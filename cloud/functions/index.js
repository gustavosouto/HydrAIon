// cloud/functions/index.js - Cloud Function para receber dados LoRa

const functions = require('firebase-functions');
const admin = require('firebase-admin');

// Inicialize o Firebase Admin SDK
// Certifique-se de que o arquivo de credenciais (serviceAccountKey.json) esteja na raiz do diretório 'functions'
// ou configure as credenciais de ambiente no Firebase.
// Para deploy no Firebase, as credenciais são automaticamente gerenciadas.
admin.initializeApp();

const db = admin.database();

/**
 * Cloud Function para receber dados de sensores via LoRa (simulado).
 * Esta função será acionada por uma requisição HTTP (POST).
 * Os dados esperados no corpo da requisição são um JSON com os valores dos sensores.
 * Exemplo de corpo da requisição:
 * {
 *   "deviceId": "sensor_001",
 *   "timestamp": 1678886400000, // Unix timestamp em milissegundos
 *   "ph": 7.2,
 *   "turbidity": 15.5,
 *   "dissolvedOxygen": 8.1,
 *   "conductivity": 350.0,
 *   "temperature": 24.8
 * }
 */
exports.receiveSensorData = functions.https.onRequest(async (req, res) => {
  // Apenas aceita requisições POST
  if (req.method !== 'POST') {
    return res.status(405).send('Method Not Allowed');
  }

  const data = req.body;

  // Validação básica dos dados
  if (!data || !data.deviceId || !data.timestamp) {
    return res.status(400).send('Dados inválidos. deviceId e timestamp são obrigatórios.');
  }

  const { deviceId, timestamp, ...sensorReadings } = data;

  try {
    // Salva os dados no Realtime Database
    // Caminho: /devices/{deviceId}/readings/{timestamp}
    await db.ref(`devices/${deviceId}/readings/${timestamp}`).set(sensorReadings);

    console.log(`Dados recebidos e salvos para o dispositivo ${deviceId} em ${new Date(timestamp).toISOString()}`);
    return res.status(200).send('Dados recebidos e salvos com sucesso!');
  } catch (error) {
    console.error('Erro ao salvar dados no Firebase:', error);
    return res.status(500).send('Erro interno do servidor ao salvar dados.');
  }
});




/**
 * Cloud Function para implementar a lógica de validação cooperativa de dados.
 * Esta função pode ser acionada após a escrita de dados no Realtime Database.
 * Ela buscará dados de sensores próximos e aplicará um algoritmo de consenso.
 * Para simplificar, vamos simular a validação com base em um limiar de desvio.
 * Em um cenário real, isso envolveria a identificação de sensores vizinhos (talvez por geolocalização)
 * e a comparação de suas leituras.
 */
exports.validateSensorData = functions.database.ref(
  '/devices/{deviceId}/readings/{timestamp}'
).onCreate(async (snapshot, context) => {
  const newSensorData = snapshot.val();
  const deviceId = context.params.deviceId;
  const timestamp = context.params.timestamp;

  console.log(`Iniciando validação para ${deviceId} em ${timestamp}`);

  // Em um cenário real, você buscaria dados de sensores próximos.
  // Por simplicidade, vamos simular que temos um "valor de referência" ou que comparamos com a média histórica.
  // Para este exemplo, vamos apenas verificar se o pH está dentro de uma faixa razoável.
  const phValue = newSensorData.ph;

  if (phValue < 6.0 || phValue > 8.5) { // Exemplo de regra de validação para pH
    console.warn(`Alerta de validação para ${deviceId}: pH (${phValue}) fora da faixa esperada.`);
    // Aqui você poderia:
    // 1. Marcar o dado como "suspeito" no banco de dados.
    // 2. Enviar um alerta para o Telegram.
    // 3. Acionar um processo de re-calibração ou verificação.
    await db.ref(`devices/${deviceId}/readings/${timestamp}/validationStatus`).set("suspect");
  } else {
    console.log(`Validação bem-sucedida para ${deviceId}: pH (${phValue}) dentro da faixa.`);
    await db.ref(`devices/${deviceId}/readings/${timestamp}/validationStatus`).set("valid");
  }

  return null; // Indica que a função foi concluída com sucesso
});




/**
 * Cloud Function para integrar com a API HidroWebService da ANA.
 * Esta função pode ser acionada para buscar dados históricos de estações da ANA
 * e enriquecer os dados do HydrAIon ou fornecer contexto.
 * 
 * Exemplo: Buscar inventário de estações telemétricas ou séries de qualidade de água.
 * 
 * Para usar esta API, geralmente não é necessária autenticação OAuth para consultas públicas,
 * mas é bom verificar a documentação mais recente da ANA.
 */
exports.fetchAnaData = functions.https.onRequest(async (req, res) => {
  if (req.method !== 'GET') {
    return res.status(405).send('Method Not Allowed');
  }

  const baseUrl = "https://www.ana.gov.br/hidrowebservice/rest/api/v1.0/EstacoesTelemetricas";
  const endpoint = "/HidroinfoanaInventarioEstacoes"; // Exemplo: Inventário de estações
  
  // Parâmetros de exemplo para a requisição (ajustar conforme a necessidade)
  // Para o inventário, pode-se usar filtros como UF, CodEstacao, etc.
  // Para séries de dados, seriam CodEstacao, dataInicio, dataFim.
  const queryParams = new URLSearchParams({
    // Exemplo: 'uf': 'PE' // Filtrar por Pernambuco
  }).toString();

  const url = `${baseUrl}${endpoint}?${queryParams}`;

  try {
    const apiResponse = await fetch(url);
    const data = await apiResponse.json();

    console.log(`Dados da ANA (${endpoint}) obtidos com sucesso.`);
    return res.status(200).json(data);
  } catch (error) {
    console.error('Erro ao buscar dados da ANA:', error);
    return res.status(500).send('Erro interno do servidor ao buscar dados da ANA.');
  }
});




/**
 * Cloud Function para acionar o re-treinamento de modelos de IA.
 * Esta função seria acionada manualmente, por um cron job, ou por um volume significativo de novos dados.
 * Em um cenário real, ela orquestraria um pipeline de ML (e.g., no Google Cloud AI Platform ou similar)
 * para re-treinar os modelos com os dados mais recentes.
 * 
 * Por ser um placeholder, ela apenas registrará o acionamento.
 */
exports.retrainAIModel = functions.https.onRequest(async (req, res) => {
  if (req.method !== 'POST') {
    return res.status(405).send('Method Not Allowed');
  }

  console.log('Acionando o re-treinamento do modelo de IA...');
  // Em um cenário real, aqui seria o código para:
  // 1. Coletar os dados mais recentes do Realtime Database ou de um bucket de armazenamento.
  // 2. Acionar um serviço de ML (e.g., Google Cloud AI Platform, Vertex AI) para re-treinar o modelo.
  // 3. Após o re-treinamento, o novo modelo seria convertido para .tflite e disponibilizado para OTA.

  return res.status(200).send('Solicitação de re-treinamento de IA recebida. Processamento em segundo plano.');
});


