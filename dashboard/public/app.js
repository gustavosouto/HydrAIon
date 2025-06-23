// Configuração do Firebase
// IMPORTANTE: Substitua estas configurações pelas suas próprias configurações do Firebase
const firebaseConfig = {
    apiKey: "SUA_API_KEY_AQUI",
    authDomain: "seu-projeto.firebaseapp.com",
    databaseURL: "https://seu-projeto-default-rtdb.firebaseio.com/",
    projectId: "seu-projeto",
    storageBucket: "seu-projeto.appspot.com",
    messagingSenderId: "123456789",
    appId: "1:123456789:web:abcdef123456"
};

// Inicializar Firebase
firebase.initializeApp(firebaseConfig);
const database = firebase.database();

// Variáveis globais
let map;
let sensorMarkers = {};

// Coordenadas do Canal do Cavouco, Recife/PE (aproximadas)
const CANAL_CAVOUCO_COORDS = [-8.0476, -34.8770];

// Inicializar o dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeMap();
    loadSensorData();
    
    // Atualizar dados a cada 30 segundos
    setInterval(loadSensorData, 30000);
});

/**
 * Inicializar o mapa Leaflet
 */
function initializeMap() {
    // Criar o mapa centrado no Canal do Cavouco
    map = L.map('map').setView(CANAL_CAVOUCO_COORDS, 15);

    // Adicionar camada de tiles do OpenStreetMap
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    // Adicionar marcador para o Canal do Cavouco
    L.marker(CANAL_CAVOUCO_COORDS)
        .addTo(map)
        .bindPopup('<b>Canal do Cavouco</b><br>Local de instalação do protótipo HydrAIon')
        .openPopup();
}

/**
 * Carregar dados dos sensores do Firebase
 */
function loadSensorData() {
    const devicesRef = database.ref('devices');
    
    devicesRef.once('value')
        .then(snapshot => {
            const devices = snapshot.val();
            if (devices) {
                updateDashboard(devices);
            } else {
                showNoDataMessage();
            }
        })
        .catch(error => {
            console.error('Erro ao carregar dados do Firebase:', error);
            showErrorMessage();
        });
}

/**
 * Atualizar o dashboard com os dados dos sensores
 */
function updateDashboard(devices) {
    const deviceIds = Object.keys(devices);
    const activeSensors = deviceIds.length;
    
    // Atualizar contadores
    document.getElementById('active-sensors').textContent = activeSensors;
    document.getElementById('last-update').textContent = new Date().toLocaleTimeString('pt-BR');
    
    // Calcular qualidade geral
    const overallQuality = calculateOverallQuality(devices);
    const qualityElement = document.getElementById('overall-quality');
    qualityElement.textContent = overallQuality.label;
    qualityElement.className = `status-value ${overallQuality.class}`;
    
    // Atualizar lista de sensores
    updateSensorList(devices);
    
    // Atualizar marcadores no mapa
    updateMapMarkers(devices);
}

/**
 * Calcular a qualidade geral da água
 */
function calculateOverallQuality(devices) {
    const deviceIds = Object.keys(devices);
    let totalScore = 0;
    let validReadings = 0;
    
    deviceIds.forEach(deviceId => {
        const device = devices[deviceId];
        if (device.readings) {
            const latestTimestamp = Math.max(...Object.keys(device.readings).map(Number));
            const latestReading = device.readings[latestTimestamp];
            
            if (latestReading && latestReading.validationStatus !== 'suspect') {
                const quality = assessWaterQuality(latestReading);
                totalScore += quality.score;
                validReadings++;
            }
        }
    });
    
    if (validReadings === 0) {
        return { label: 'Sem Dados', class: 'warning' };
    }
    
    const averageScore = totalScore / validReadings;
    
    if (averageScore >= 80) {
        return { label: 'Boa', class: 'good' };
    } else if (averageScore >= 60) {
        return { label: 'Regular', class: 'warning' };
    } else {
        return { label: 'Crítica', class: 'danger' };
    }
}

/**
 * Avaliar a qualidade da água com base nos parâmetros
 */
function assessWaterQuality(reading) {
    let score = 100;
    let status = 'good';
    
    // Avaliar pH (ideal: 6.5 - 8.5)
    if (reading.ph) {
        if (reading.ph < 6.0 || reading.ph > 9.0) {
            score -= 30;
            status = 'danger';
        } else if (reading.ph < 6.5 || reading.ph > 8.5) {
            score -= 15;
            if (status === 'good') status = 'warning';
        }
    }
    
    // Avaliar turbidez (ideal: < 5 NTU)
    if (reading.turbidity) {
        if (reading.turbidity > 20) {
            score -= 25;
            status = 'danger';
        } else if (reading.turbidity > 5) {
            score -= 10;
            if (status === 'good') status = 'warning';
        }
    }
    
    // Avaliar oxigênio dissolvido (ideal: > 5 mg/L)
    if (reading.dissolvedOxygen) {
        if (reading.dissolvedOxygen < 3) {
            score -= 25;
            status = 'danger';
        } else if (reading.dissolvedOxygen < 5) {
            score -= 10;
            if (status === 'good') status = 'warning';
        }
    }
    
    // Avaliar condutividade (ideal: 50-500 µS/cm para água doce)
    if (reading.conductivity) {
        if (reading.conductivity > 1000 || reading.conductivity < 10) {
            score -= 20;
            if (status !== 'danger') status = 'warning';
        }
    }
    
    return { score: Math.max(0, score), status };
}

/**
 * Atualizar a lista de sensores
 */
function updateSensorList(devices) {
    const sensorList = document.getElementById('sensor-list');
    sensorList.innerHTML = '';
    
    Object.keys(devices).forEach(deviceId => {
        const device = devices[deviceId];
        if (device.readings) {
            const latestTimestamp = Math.max(...Object.keys(device.readings).map(Number));
            const latestReading = device.readings[latestTimestamp];
            
            const sensorElement = createSensorElement(deviceId, latestReading, latestTimestamp);
            sensorList.appendChild(sensorElement);
        }
    });
}

/**
 * Criar elemento HTML para um sensor
 */
function createSensorElement(deviceId, reading, timestamp) {
    const quality = assessWaterQuality(reading);
    const div = document.createElement('div');
    div.className = `sensor-item ${quality.status}`;
    
    const statusLabels = {
        'good': 'Boa',
        'warning': 'Regular',
        'danger': 'Crítica'
    };
    
    div.innerHTML = `
        <div class="sensor-header">
            <span class="sensor-id">${deviceId}</span>
            <span class="sensor-status ${quality.status}">${statusLabels[quality.status]}</span>
        </div>
        <div class="sensor-readings">
            ${reading.ph ? `
                <div class="reading">
                    <div class="reading-label">pH</div>
                    <div class="reading-value">${reading.ph.toFixed(1)}</div>
                </div>
            ` : ''}
            ${reading.turbidity ? `
                <div class="reading">
                    <div class="reading-label">Turbidez</div>
                    <div class="reading-value">${reading.turbidity.toFixed(1)} NTU</div>
                </div>
            ` : ''}
            ${reading.dissolvedOxygen ? `
                <div class="reading">
                    <div class="reading-label">OD</div>
                    <div class="reading-value">${reading.dissolvedOxygen.toFixed(1)} mg/L</div>
                </div>
            ` : ''}
            ${reading.conductivity ? `
                <div class="reading">
                    <div class="reading-label">Condutividade</div>
                    <div class="reading-value">${reading.conductivity.toFixed(0)} µS/cm</div>
                </div>
            ` : ''}
            ${reading.temperature ? `
                <div class="reading">
                    <div class="reading-label">Temperatura</div>
                    <div class="reading-value">${reading.temperature.toFixed(1)}°C</div>
                </div>
            ` : ''}
        </div>
        <div style="margin-top: 1rem; font-size: 0.8rem; color: #666;">
            Última leitura: ${new Date(timestamp).toLocaleString('pt-BR')}
        </div>
    `;
    
    return div;
}

/**
 * Atualizar marcadores no mapa
 */
function updateMapMarkers(devices) {
    // Para este exemplo, vamos posicionar os sensores em pontos próximos ao Canal do Cavouco
    // Em um cenário real, cada sensor teria suas próprias coordenadas GPS
    const baseCoords = CANAL_CAVOUCO_COORDS;
    let deviceIndex = 0;
    
    Object.keys(devices).forEach(deviceId => {
        const device = devices[deviceId];
        if (device.readings) {
            const latestTimestamp = Math.max(...Object.keys(device.readings).map(Number));
            const latestReading = device.readings[latestTimestamp];
            const quality = assessWaterQuality(latestReading);
            
            // Posicionar sensores em um pequeno raio ao redor do canal
            const offset = 0.002; // Aproximadamente 200 metros
            const angle = (deviceIndex * 60) * (Math.PI / 180); // Distribuir em círculo
            const lat = baseCoords[0] + (offset * Math.cos(angle));
            const lng = baseCoords[1] + (offset * Math.sin(angle));
            
            // Cores dos marcadores baseadas na qualidade
            const markerColors = {
                'good': 'green',
                'warning': 'orange',
                'danger': 'red'
            };
            
            // Remover marcador existente se houver
            if (sensorMarkers[deviceId]) {
                map.removeLayer(sensorMarkers[deviceId]);
            }
            
            // Criar novo marcador
            const marker = L.circleMarker([lat, lng], {
                color: markerColors[quality.status],
                fillColor: markerColors[quality.status],
                fillOpacity: 0.7,
                radius: 8
            }).addTo(map);
            
            // Popup com informações do sensor
            const popupContent = `
                <b>${deviceId}</b><br>
                Status: ${quality.status === 'good' ? 'Boa' : quality.status === 'warning' ? 'Regular' : 'Crítica'}<br>
                ${latestReading.ph ? `pH: ${latestReading.ph.toFixed(1)}<br>` : ''}
                ${latestReading.turbidity ? `Turbidez: ${latestReading.turbidity.toFixed(1)} NTU<br>` : ''}
                ${latestReading.dissolvedOxygen ? `OD: ${latestReading.dissolvedOxygen.toFixed(1)} mg/L<br>` : ''}
                ${latestReading.temperature ? `Temp: ${latestReading.temperature.toFixed(1)}°C<br>` : ''}
                <small>Atualizado: ${new Date(latestTimestamp).toLocaleTimeString('pt-BR')}</small>
            `;
            
            marker.bindPopup(popupContent);
            sensorMarkers[deviceId] = marker;
            deviceIndex++;
        }
    });
}

/**
 * Mostrar mensagem quando não há dados
 */
function showNoDataMessage() {
    document.getElementById('active-sensors').textContent = '0';
    document.getElementById('last-update').textContent = 'Nunca';
    document.getElementById('overall-quality').textContent = 'Sem Dados';
    
    const sensorList = document.getElementById('sensor-list');
    sensorList.innerHTML = '<p style="text-align: center; color: #666; padding: 2rem;">Nenhum dado de sensor disponível. Verifique se os dispositivos estão conectados.</p>';
}

/**
 * Mostrar mensagem de erro
 */
function showErrorMessage() {
    document.getElementById('active-sensors').textContent = 'Erro';
    document.getElementById('last-update').textContent = 'Erro';
    document.getElementById('overall-quality').textContent = 'Erro';
    
    const sensorList = document.getElementById('sensor-list');
    sensorList.innerHTML = '<p style="text-align: center; color: #dc3545; padding: 2rem;">Erro ao carregar dados. Verifique a conexão com o Firebase.</p>';
}

