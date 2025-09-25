/**
 * @file esp32_uwb_main.cpp
 * @author Your Name
 * @brief Firmware for an ESP32-based UWB (Ultra-Wideband) device with IMU and Barometer.
 * @version 0.1
 * @date 2023-10-27
 *
 * @copyright Copyright (c) 2023
 *
 * This firmware enables an ESP32 to act as either a UWB Tag or Anchor for
 * real-time location systems. It reads sensor data from an MPU6050 (IMU) and a
 * BMP388 (barometer), combines it with UWB range data, and sends it over UDP
 * to a central server. The device's role (Tag or Anchor) can be changed
 * dynamically via UDP commands.
 *
 * The system uses FreeRTOS to handle UWB communication and network operations in
 * separate, parallel tasks for improved stability and performance.
 *
 * Core Components:
 * - ESP32 WiFi for network communication.
 * - DW1000 for UWB ranging.
 * - MPU6050 for inertial measurements (accelerometer, gyroscope).
 * - BMP388 for barometric pressure readings.
 * - FreeRTOS for multitasking.
 *
 * Pinout:
 * - MPU6050 & BMP388: I2C (SDA, SCL)
 * - DW1000: SPI (MOSI, MISO, SCK, CS), plus RST and IRQ pins.
 */

// =========================================================
// INCLUDES
// =========================================================
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <SPI.h>
#include "Adafruit_BMP3XX.h" // Library for the BMP388 sensor
#include "DW1000Ranging.h"   // DW1000 UWB library
#include "DW1000.h"

// =========================================================
// CONFIGURATION
// =========================================================

// --- WiFi & Network Settings ---
const char* ssid = "Phoenix";               // Your WiFi network SSID
const char* password = "123456789";         // Your WiFi network password
const char* udpAddress = "10.21.65.119";    // IP address of the server to send data to
const int udpPort = 1234;                   // Port on the server to send data to
const int udpListenPort = 1235;             // Port on this ESP32 to listen for commands
WiFiUDP udpSend;                            // UDP object for sending data
WiFiUDP udpReceive;                         // UDP object for receiving commands

// --- Device Specific Settings ---
// !!! IMPORTANT: CHANGE THIS FOR EACH DEVICE to ensure unique identification !!!
char MY_ADDRESS[] = "01:00:00:00:00:00:00:01";

// --- Sensor & Hardware Pins ---
const int MPU_ADDR = 0x68;      // I2C address of the MPU6050 IMU
Adafruit_BMP3XX bmp;            // BMP388 sensor object
const uint8_t PIN_RST = 27;     // DW1000 Reset pin
const uint8_t PIN_IRQ = 34;     // DW1000 Interrupt Request pin
const uint8_t PIN_SS  = 5;      // DW1000 SPI Slave Select pin

// --- System State ---
// Defines the possible roles for the UWB device
enum Role { ROLE_TAG, ROLE_ANCHOR, ROLE_IDLE };
// Holds the current role of the device, volatile as it can be changed by an interrupt/callback
volatile Role currentRole = ROLE_IDLE;

// --- FreeRTOS Handles, Queue, and Mutex ---
TaskHandle_t UWBTask;       // Handle for the UWB processing task
TaskHandle_t NetworkTask;   // Handle for the network communication task
QueueHandle_t dataQueue;    // Queue to pass data from UWB/sensor callbacks to the network task
SemaphoreHandle_t uwbMutex; // Mutex to protect access to the DW1000 module from concurrent tasks

// --- Data Structure ---
// Struct to hold a complete data packet from all sensors and UWB
struct DataPacket {
  uint16_t remoteAddress; // Short address of the other UWB device
  float ax, ay, az;       // Accelerometer data
  float gx, gy, gz;       // Gyroscope data
  float pressure;         // Barometric pressure
  float uwb_range;        // UWB range measurement
};

// =========================================================
// SETUP
// =========================================================
void setup() {
  Serial.begin(115200);
  delay(1000); // Wait for serial to initialize

  // --- Initialize FreeRTOS components ---
  dataQueue = xQueueCreate(10, sizeof(DataPacket)); // Create a queue for 10 data packets
  uwbMutex = xSemaphoreCreateMutex();               // Create a mutex for UWB hardware access

  // --- WiFi Connection ---
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println(" Connected!");
  Serial.print("ESP32 IP Address: "); Serial.println(WiFi.localIP());

  // --- Initialize I2C Sensors ---
  Wire.begin();
  // Wake up MPU6050
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); // PWR_MGMT_1 register
  Wire.write(0);    // Set to zero (wakes up the MPU-6050)
  Wire.endTransmission(true);
  Serial.println("MPU OK.");

  // Initialize BMP388
  if (!bmp.begin_I2C()) {
    Serial.println("BMP388 not found. Check wiring.");
  } else {
    // Set sensor settings for good performance
    bmp.setTemperatureOversampling(BMP3_OVERSAMPLING_8X);
    bmp.setPressureOversampling(BMP3_OVERSAMPLING_4X);
    bmp.setIIRFilterCoeff(BMP3_IIR_FILTER_COEFF_3);
    Serial.println("BMP388 OK.");
  }

  // --- Initialize DW1000 ---
  DW1000Ranging.initCommunication(PIN_RST, PIN_SS, PIN_IRQ);
  // Register callbacks for UWB events
  DW1000Ranging.attachNewRange(newRange);
  DW1000Ranging.attachNewDevice(newDevice);
  DW1000Ranging.attachBlinkDevice(newBlink);

  Serial.println("Node initialized. Creating tasks...");

  // --- Create FreeRTOS Tasks ---
  // Pin UWB task to Core 1 and Network task to Core 0 to prevent conflicts
  xTaskCreatePinnedToCore(uwbLoop, "UWBTask", 10000, NULL, 2, &UWBTask, 1);
  xTaskCreatePinnedToCore(networkLoop, "NetworkTask", 10000, NULL, 1, &NetworkTask, 0);

  Serial.println("Setup complete. Waiting for role command from server...");
}

/**
 * @brief Main loop is empty because all work is done in FreeRTOS tasks.
 */
void loop() {
  // Delay to yield time to other tasks, especially the idle task.
  vTaskDelay(1000 / portTICK_PERIOD_MS);
}

// =========================================================
// --- FreeRTOS TASKS ---
// =========================================================

/**
 * @brief Task dedicated to polling the DW1000 module.
 *
 * This task continuously calls the DW1000Ranging.loop() function, which is
 * necessary for the UWB module to operate. It uses a mutex to ensure that
 * role-switching does not happen in the middle of a UWB operation.
 * @param pvParameters Not used.
 */
void uwbLoop(void * pvParameters) {
  Serial.println("UWB Task started on Core 1");
  for (;;) {
    // Take the mutex to ensure exclusive access to the DW1000 hardware
    if (xSemaphoreTake(uwbMutex, portMAX_DELAY) == pdTRUE) {
      DW1000Ranging.loop(); // Main polling function for the UWB library
      // Give the mutex back once done
      xSemaphoreGive(uwbMutex);
    }
    // A small delay to prevent this task from starving other lower-priority tasks
    vTaskDelay(1 / portTICK_PERIOD_MS);
  }
}

/**
 * @brief Task for handling all network communication.
 *
 * This task checks for incoming UDP commands and sends collected sensor data
 * from the queue to the central server.
 * @param pvParameters Not used.
 */
void networkLoop(void * pvParameters) {
  Serial.println("Network Task started on Core 0");
  udpReceive.begin(udpListenPort); // Start listening for UDP packets
  DataPacket receivedPacket;

  for (;;) {
    // Check for any commands (e.g., role change) from the server
    checkUDPCommands();

    // Check if there is a new data packet in the queue to be sent
    if (xQueueReceive(dataQueue, &receivedPacket, (TickType_t)0) == pdPASS) {
      char dataString[256];
      // Format the data into a comma-separated string
      sprintf(dataString, "%s,%X,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f",
              MY_ADDRESS, receivedPacket.remoteAddress,
              receivedPacket.ax, receivedPacket.ay, receivedPacket.az,
              receivedPacket.gx, receivedPacket.gy, receivedPacket.gz,
              receivedPacket.pressure, receivedPacket.uwb_range);

      // Send the data string via UDP
      udpSend.beginPacket(udpAddress, udpPort);
      udpSend.printf(dataString);
      udpSend.endPacket();

      // Serial.println(dataString); // Uncomment for local debugging
    }
    // Delay to allow other tasks to run
    vTaskDelay(10 / portTICK_PERIOD_MS);
  }
}

// =========================================================
// COMMAND AND ROLE MANAGEMENT
// =========================================================

/**
 * @brief Parses incoming UDP packets for commands.
 *
 * Currently supports the "ROLE:" command to switch between TAG and ANCHOR modes.
 * Example command: "ROLE:TAG" or "ROLE:ANCHOR"
 */
void checkUDPCommands() {
  int packetSize = udpReceive.parsePacket();
  if (packetSize) {
    char packetBuffer[32];
    int len = udpReceive.read(packetBuffer, 31); // Read the packet into a buffer
    if (len > 0) {
      packetBuffer[len] = 0; // Null-terminate the string
      String command = String(packetBuffer);
      if (command.startsWith("ROLE:")) {
        String newRoleStr = command.substring(5);
        if (newRoleStr == "TAG" && currentRole != ROLE_TAG) {
            switchRole(ROLE_TAG);
        } else if (newRoleStr == "ANCHOR" && currentRole != ROLE_ANCHOR) {
            switchRole(ROLE_ANCHOR);
        }
      }
    }
  }
}

/**
 * @brief Switches the UWB device role between ANCHOR and TAG.
 *
 * This function stops the current UWB operation, resets the DW1000 module,
 * and restarts it in the new specified role. This entire operation is protected
 * by a mutex to prevent race conditions with the uwbLoop task.
 * @param newRole The new role to switch to (ROLE_TAG or ROLE_ANCHOR).
 */
void switchRole(Role newRole) {
  // Take the mutex to ensure the UWB task is not using the hardware
  if (xSemaphoreTake(uwbMutex, portMAX_DELAY) == pdTRUE) {
    // Reset the DW1000 to ensure a clean state before switching roles
    DW1000.reset();
    vTaskDelay(50 / portTICK_PERIOD_MS); // Small delay to let hardware settle

    if (newRole == ROLE_ANCHOR) {
      Serial.println("Command received: Switching to ANCHOR mode...");
      // Start the device as an anchor with its unique address
      DW1000Ranging.startAsAnchor(MY_ADDRESS, DW1000.MODE_LONGDATA_RANGE_ACCURACY);
    } else if (newRole == ROLE_TAG) {
      Serial.println("Command received: Switching to TAG mode...");
      // Start the device as a tag, broadcasting to all anchors
      DW1000Ranging.startAsTag("FF:FF:FF:FF:FF:FF", DW1000.MODE_LONGDATA_RANGE_ACCURACY);
    }
    currentRole = newRole;

    // Release the mutex so the UWB task can resume
    xSemaphoreGive(uwbMutex);
  }
}

// =========================================================
// UWB AND SENSOR CALLBACKS
// =========================================================

/**
 * @brief Callback function executed when a new UWB range is calculated.
 *
 * This function is the primary data acquisition point. It reads the UWB range,
 * polls the MPU6050 and BMP388 for the latest sensor data, packs it all into a
 * DataPacket struct, and sends it to the data queue for the network task.
 */
void newRange() {
  DataPacket packet;
  // Get UWB data
  packet.remoteAddress = DW1000Ranging.getDistantDevice()->getShortAddress();
  packet.uwb_range = DW1000Ranging.getDistantDevice()->getRange();

  // --- Read MPU6050 IMU Data ---
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B); // Starting register for accelerometer data
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, true); // Request 14 bytes (Accel, Temp, Gyro)

  // Read raw sensor values
  int16_t ax_raw = Wire.read() << 8 | Wire.read();
  int16_t ay_raw = Wire.read() << 8 | Wire.read();
  int16_t az_raw = Wire.read() << 8 | Wire.read();
  Wire.read(); Wire.read(); // Skip temperature bytes
  int16_t gx_raw = Wire.read() << 8 | Wire.read();
  int16_t gy_raw = Wire.read() << 8 | Wire.read();
  int16_t gz_raw = Wire.read() << 8 | Wire.read();

  // Convert raw values to physical units (g's and degrees/sec)
  packet.ax = ax_raw / 16384.0;
  packet.ay = ay_raw / 16384.0;
  packet.az = az_raw / 16384.0;
  packet.gx = gx_raw / 131.0;
  packet.gy = gy_raw / 131.0;
  packet.gz = gz_raw / 131.0;

  // --- Read BMP388 Barometer Data ---
  if (bmp.performReading()) {
    packet.pressure = bmp.pressure / 100.0; // Convert Pa to hPa (millibars)
  } else {
    packet.pressure = 0.0; // Indicate failure to read
  }

  // Send the populated data packet to the queue.
  // Use a non-blocking send; if the queue is full, the data is dropped.
  xQueueSend(dataQueue, &packet, (TickType_t)0);
}

/**
 * @brief Callback for when a new UWB device is detected by a Tag.
 * @param device Pointer to the detected device object.
 */
void newDevice(DW1000Device* device) {
  Serial.print("Tag saw new device: ");
  Serial.println(device->getShortAddress(), HEX);
}

/**
 * @brief Callback for when an Anchor hears a "blink" (initial broadcast) from a Tag.
 * @param device Pointer to the blinking device object.
 */
void newBlink(DW1000Device* device) {
  Serial.print("Anchor heard a blink from: ");
  Serial.println(device->getShortAddress(), HEX);
}