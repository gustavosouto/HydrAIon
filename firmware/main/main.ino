// main.ino - Esqueleto do firmware para o HydrAIon

#include <Arduino.h>

void setup() {
  // Inicialização da comunicação serial para depuração
  Serial.begin(115200);
  Serial.println("HydrAIon - Firmware Inicializado!");

  // TODO: Inicializar sensores
  // TODO: Inicializar módulo LoRa
  // TODO: Configurar gerenciamento de energia
}

void loop() {
  // TODO: Ler dados dos sensores
  // TODO: Realizar inferência TinyML
  // TODO: Enviar dados via LoRa
  // TODO: Entrar em modo deep sleep (se aplicável)

  delay(1000); // Pequeno delay para evitar loop muito rápido (ajustar conforme necessidade)
}



// Incluir a biblioteca DFRobot_PH (assumindo que será instalada via PlatformIO ou Arduino IDE)
#include <DFRobot_PH.h>

// Definir o pino analógico para o sensor de pH
#define PH_PIN A0

// Criar uma instância do sensor de pH
DFRobot_PH ph;

void setup() {
  // Inicialização da comunicação serial para depuração
  Serial.begin(115200);
  Serial.println("HydrAIon - Firmware Inicializado!");

  // Inicializar o sensor de pH
  ph.begin();

  // TODO: Inicializar outros sensores
  // TODO: Inicializar módulo LoRa
  // TODO: Configurar gerenciamento de energia
}

void loop() {
  static unsigned long timepoint = millis();
  if(millis()-timepoint > 1000U){  // Leitura a cada 1 segundo
    timepoint = millis();
    
    // Leitura do valor analógico do sensor de pH
    float voltage = analogRead(PH_PIN) / 1024.0 * 5.0; // Converter para voltagem (0-5V)
    
    // Obter o valor de pH (necessita calibração)
    // O valor de 'temperature' é importante para a compensação de temperatura do pH
    // Por enquanto, usaremos um valor fixo, mas será lido do sensor de temperatura
    float temperature = 25.0; // TODO: Substituir pela leitura real do sensor de temperatura
    float phValue = ph.readPH(voltage, temperature);

    Serial.print("Voltagem pH: ");
    Serial.print(voltage, 2);
    Serial.print("V, pH: ");
    Serial.println(phValue, 2);
  }

  // TODO: Realizar inferência TinyML
  // TODO: Enviar dados via LoRa
  // TODO: Entrar em modo deep sleep (se aplicável)

  // delay(1000); // Removido, pois a leitura é baseada em millis()
}



// Definir o pino analógico para o sensor de turbidez
#define TURBIDITY_PIN A1 // Assumindo que A1 está disponível e conectado

void setup() {
  // ... (código existente)

  // TODO: Inicializar outros sensores
  // TODO: Inicializar módulo LoRa
  // TODO: Configurar gerenciamento de energia
}

void loop() {
  static unsigned long timepoint = millis();
  if(millis()-timepoint > 1000U){  // Leitura a cada 1 segundo
    timepoint = millis();
    
    // ... (código do sensor de pH)

    // Leitura do valor analógico do sensor de turbidez
    float turbidityVoltage = analogRead(TURBIDITY_PIN) / 4095.0 * 3.3; // Converter para voltagem (0-3.3V para ESP32)
    
    // A conversão de voltagem para NTU para o SEN0189 é não-linear e depende da calibração.
    // O DFRobot fornece um gráfico de referência. Para um código simples, podemos usar uma aproximação.
    // É crucial calibrar o sensor com soluções de turbidez conhecidas para obter valores precisos.
    // Para fins de exemplo, usaremos uma função linear simples que precisará ser ajustada.
    float turbidityNTU = 0.0; // Valor inicial

    // Exemplo de mapeamento aproximado (precisa ser calibrado com dados reais)
    // Baseado na documentação do DFRobot, 4.1V ~ 0.5 NTU (água pura)
    // e a turbidez aumenta à medida que a voltagem diminui.
    // Esta é uma aproximação linear inversa para demonstração.
    if (turbidityVoltage < 2.5) { // Abaixo de 2.5V, turbidez alta
      turbidityNTU = 3000; // Valor máximo aproximado
    } else if (turbidityVoltage < 3.0) { // Entre 2.5V e 3.0V
      turbidityNTU = map(turbidityVoltage * 100, 250, 300, 3000, 1000) / 100.0; // Mapeamento inverso
    } else if (turbidityVoltage < 3.5) { // Entre 3.0V e 3.5V
      turbidityNTU = map(turbidityVoltage * 100, 300, 350, 1000, 100) / 100.0; // Mapeamento inverso
    } else if (turbidityVoltage < 4.0) { // Entre 3.5V e 4.0V
      turbidityNTU = map(turbidityVoltage * 100, 350, 400, 100, 10) / 100.0; // Mapeamento inverso
    } else { // Acima de 4.0V, turbidez muito baixa
      turbidityNTU = 0.5; // Água muito limpa
    }

    Serial.print("Voltagem Turbidez: ");
    Serial.print(turbidityVoltage, 2);
    Serial.print("V, Turbidez: ");
    Serial.print(turbidityNTU, 2);
    Serial.println(" NTU");
  }

  // TODO: Realizar inferência TinyML
  // TODO: Enviar dados via LoRa
  // TODO: Entrar em modo deep sleep (se aplicável)

}



// Incluir a biblioteca Wire para comunicação I2C
#include <Wire.h>
// Incluir a biblioteca Atlas Scientific EZO I2C (assumindo que será instalada)
#include <Ezo_i2c.h>

// Definir o endereço I2C para o sensor EZO-DO
// O endereço padrão para o EZO-DO é 0x61 (97 em decimal)
#define DO_SENSOR_ADDRESS 97

// Criar uma instância do sensor DO
Ezo_i2c do_sensor(DO_SENSOR_ADDRESS);

void setup() {
  // ... (código existente)

  // Inicializar comunicação I2C
  Wire.begin();

  // Inicializar o sensor DO
  do_sensor.begin();

  // TODO: Inicializar outros sensores (Condutividade, Temperatura)
  // TODO: Inicializar módulo LoRa
  // TODO: Configurar gerenciamento de energia
}

void loop() {
  static unsigned long timepoint = millis();
  if(millis()-timepoint > 1000U){  // Leitura a cada 1 segundo
    timepoint = millis();
    
    // ... (código do sensor de pH e turbidez)

    // Leitura do sensor de Oxigênio Dissolvido (DO)
    do_sensor.send_cmd("R"); // Envia comando de leitura
    delay(300); // Espera pela resposta (tempo recomendado pela Atlas Scientific)

    String response = do_sensor.get_response();
    if (response.startsWith("*")) { // Se a resposta começar com '*', é um erro
      Serial.print("Erro DO: ");
      Serial.println(response);
    } else {
      float dissolvedOxygen = response.toFloat();
      Serial.print("Oxigênio Dissolvido: ");
      Serial.print(dissolvedOxygen, 2);
      Serial.println(" mg/L");
    }
  }

  // TODO: Realizar inferência TinyML
  // TODO: Enviar dados via LoRa
  // TODO: Entrar em modo deep sleep (se aplicável)

}



// Definir o endereço I2C para o sensor EZO-EC
// O endereço padrão para o EZO-EC é 0x64 (100 em decimal)
#define EC_SENSOR_ADDRESS 100

// Criar uma instância do sensor EC
Ezo_i2c ec_sensor(EC_SENSOR_ADDRESS);

void setup() {
  // ... (código existente)

  // Inicializar o sensor EC
  ec_sensor.begin();

  // TODO: Inicializar sensor de Temperatura
  // TODO: Inicializar módulo LoRa
  // TODO: Configurar gerenciamento de energia
}

void loop() {
  static unsigned long timepoint = millis();
  if(millis()-timepoint > 1000U){  // Leitura a cada 1 segundo
    timepoint = millis();
    
    // ... (código do sensor de pH e turbidez e DO)

    // Leitura do sensor de Condutividade Elétrica (EC)
    ec_sensor.send_cmd("R"); // Envia comando de leitura
    delay(300); // Espera pela resposta

    String response = ec_sensor.get_response();
    if (response.startsWith("*")) { // Se a resposta começar com 
      Serial.print("Erro EC: ");
      Serial.println(response);
    } else {
      float conductivity = response.toFloat();
      Serial.print("Condutividade: ");
      Serial.print(conductivity, 2);
      Serial.println(" uS/cm");

      // Para obter Salinidade e TDS, comandos adicionais podem ser enviados
      // ec_sensor.send_cmd("S,?"); // Pergunta pela salinidade
      // ec_sensor.send_cmd("TDS,?"); // Pergunta pelo TDS
    }
  }

  // TODO: Realizar inferência TinyML
  // TODO: Enviar dados via LoRa
  // TODO: Entrar em modo deep sleep (se aplicável)

}



// Incluir as bibliotecas para o sensor DS18B20
#include <OneWire.h>
#include <DallasTemperature.h>

// Definir o pino OneWire para o DS18B20
#define ONE_WIRE_BUS 4 // Exemplo: Pino GPIO 4 do ESP32

// Configurar uma instância OneWire para se comunicar com qualquer dispositivo OneWire
OneWire oneWire(ONE_WIRE_BUS);

// Passar a referência OneWire para a biblioteca Dallas Temperature
DallasTemperature sensors(&oneWire);

void setup() {
  // ... (código existente)

  // Inicializar o sensor de temperatura DS18B20
  sensors.begin();

  // TODO: Inicializar módulo LoRa
  // TODO: Configurar gerenciamento de energia
}

void loop() {
  static unsigned long timepoint = millis();
  if(millis()-timepoint > 1000U){  // Leitura a cada 1 segundo
    timepoint = millis();
    
    // ... (código do sensor de pH, turbidez e DO)

    // Leitura do sensor de Temperatura DS18B20
    sensors.requestTemperatures(); // Envia o comando para todos os sensores na rede OneWire
    float temperatureC = sensors.getTempCByIndex(0); // Obtém a temperatura do primeiro sensor

    if (temperatureC != DEVICE_DISCONNECTED_C) {
      Serial.print("Temperatura: ");
      Serial.print(temperatureC);
      Serial.println(" °C");

      // Atualizar a temperatura usada na compensação do pH
      // ph.readPH(voltage, temperatureC);
    } else {
      Serial.println("Erro: Sensor de temperatura DS18B20 desconectado ou com falha.");
    }
  }

  // TODO: Realizar inferência TinyML
  // TODO: Enviar dados via LoRa
  // TODO: Entrar em modo deep sleep (se aplicável)

}



// Incluir as bibliotecas TensorFlow Lite para Microcontrollers
// Certifique-se de que estas bibliotecas estão instaladas no seu ambiente Arduino IDE/PlatformIO
// Ex: #include <tensorflow/lite/micro/all_ops_resolver.h>
// Ex: #include <tensorflow/lite/micro/micro_error_reporter.h>
// Ex: #include <tensorflow/lite/micro/micro_interpreter.h>
// Ex: #include <tensorflow/lite/schema/schema_generated.h>
// Ex: #include <tensorflow/lite/version.h>

// Placeholder para o modelo TensorFlow Lite (será gerado na Fase 4)
// const unsigned char model_data[] = { ... };

// Tamanho da arena de tensores (ajustar conforme a complexidade do modelo)
const int kTensorArenaSize = 8 * 1024; // 8 KB, pode precisar de mais
uint8_t tensor_arena[kTensorArenaSize];

// Ponteiros para o interpretador e modelo
tflite::MicroErrorReporter tflErrorReporter;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteModel* tflModel = nullptr;

void setup() {
  // ... (código existente)

  // TODO: Inicializar módulo LoRa
  // TODO: Configurar gerenciamento de energia

  // Inicializar o modelo TinyML (placeholder)
  // if (tflModel == nullptr) {
  //   tflModel = tflite::GetModel(model_data);
  //   if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
  //     Serial.println("Erro: Versão do modelo incompatível!");
  //     return;
  //   }
  // }

  // tflite::AllOpsResolver resolver;
  // tflInterpreter = new tflite::MicroInterpreter(
  //     tflModel, resolver, tensor_arena, kTensorArenaSize, &tflErrorReporter);

  // if (tflInterpreter->AllocateTensors() != kTfLiteOk) {
  //   Serial.println("Erro: Falha ao alocar tensores!");
  //   return;
  // }
  // Serial.println("TinyML: Modelo carregado e tensores alocados.");
}

void loop() {
  static unsigned long timepoint = millis();
  if(millis()-timepoint > 1000U){  // Leitura a cada 1 segundo
    timepoint = millis();
    
    // ... (código dos sensores)

    // Exemplo de inferência TinyML (placeholder)
    // if (tflInterpreter != nullptr) {
    //   TfLiteTensor* input = tflInterpreter->input(0);
    //   // Preencher o input com dados dos sensores
    //   // input->data.f[0] = phValue;
    //   // input->data.f[1] = turbidityNTU;
    //   // ...

    //   if (tflInterpreter->Invoke() != kTfLiteOk) {
    //     Serial.println("Erro: Falha na inferência TinyML!");
    //     return;
    //   }

    //   TfLiteTensor* output = tflInterpreter->output(0);
    //   // Processar o output do modelo
    //   // float prediction = output->data.f[0];
    //   // Serial.print("TinyML Predição: ");
    //   // Serial.println(prediction);
    // }
  }

  // TODO: Enviar dados via LoRa
  // TODO: Entrar em modo deep sleep (se aplicável)

}



// Incluir a biblioteca RadioLib para LoRa
#include <RadioLib.h>

// Definir os pinos LoRa (estes pinos podem variar dependendo da sua placa ESP32 LoRa)
// Exemplo para TTGO LoRa32 V2.0/V2.1
#define LORA_CS 18
#define LORA_RST 14
#define LORA_DIO0 26
#define LORA_DIO1 33
#define LORA_DIO2 32

// Criar uma instância da classe SX1276 (ou SX1278, dependendo do módulo LoRa)
// SX1276 radio = new Module(LORA_CS, LORA_DIO0, LORA_RST, LORA_DIO1, LORA_DIO2);
// Para ESP32, geralmente o SPI é configurado automaticamente, mas pode ser necessário especificar.
// Se estiver usando uma placa como TTGO LoRa32, os pinos SPI (SCK, MISO, MOSI) já estão conectados internamente.
SX1276 radio = new Module(LORA_CS, LORA_DIO0, LORA_RST, LORA_DIO1);

void setup() {
  // ... (código existente)

  // Inicializar o módulo LoRa
  Serial.print("[LoRa] Inicializando...");
  int state = radio.begin(433.0); // Frequência LoRa (ex: 433.0 MHz, 868.0 MHz, 915.0 MHz)

  if (state == RADIOLIB_ERR_NONE) {
    Serial.println("Sucesso!");
  } else {
    Serial.print("Falha, código: ");
    Serial.println(state);
    while(true); // Trava se o LoRa não inicializar
  }

  // TODO: Configurar gerenciamento de energia
}

void loop() {
  static unsigned long timepoint = millis();
  if(millis()-timepoint > 1000U){  // Leitura a cada 1 segundo
    timepoint = millis();
    
    // ... (código dos sensores)

    // Exemplo de envio de dados via LoRa
    String message = "pH: " + String(phValue, 2) + ", Turb: " + String(turbidityNTU, 2) + ", DO: " + String(dissolvedOxygen, 2) + ", Temp: " + String(temperatureC, 2);
    Serial.print("[LoRa] Enviando pacote: ");
    Serial.println(message);

    int state = radio.transmit(message);

    if (state == RADIOLIB_ERR_NONE) {
      Serial.println("[LoRa] Pacote enviado com sucesso!");
    } else if (state == RADIOLIB_ERR_TX_TIMEOUT) {
      Serial.println("[LoRa] Tempo limite de transmissão excedido!");
    } else {
      Serial.print("[LoRa] Falha na transmissão, código: ");
      Serial.println(state);
    }
  }

  // TODO: Entrar em modo deep sleep (se aplicável)

}



// Incluir a biblioteca para gerenciamento de energia do ESP32
#include <esp_sleep.h>

// Definir o tempo de deep sleep em microssegundos (ex: 5 minutos = 300 segundos = 300 * 1000000 microssegundos)
#define DEEP_SLEEP_TIME_S 300 // 5 minutos

void setup() {
  // ... (código existente)

  // Configurar o timer para acordar do deep sleep
  esp_sleep_enable_timer_wakeup(DEEP_SLEEP_TIME_S * 1000000ULL);

  Serial.println("Configurando ESP32 para Deep Sleep...");
}

void loop() {
  static unsigned long timepoint = millis();
  if(millis()-timepoint > 1000U){  // Leitura a cada 1 segundo
    timepoint = millis();
    
    // ... (código dos sensores e LoRa)

    // Após enviar os dados, entrar em deep sleep
    Serial.println("Entrando em Deep Sleep...");
    Serial.flush(); // Garante que todas as mensagens seriais sejam enviadas antes de dormir
    esp_deep_sleep_start();
  }

}

