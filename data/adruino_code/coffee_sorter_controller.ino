// Enhanced Coffee Sorter - Arduino Controller
// ระบบคัดแยกเมล็ดกาแฟด้วย AI Vision + Pneumatic Cylinder

// --- การตั้งค่าฮาร์ดแวร์ (Hardware Configuration) ---
const int CYLINDER_RELAY_PIN = 4;     // รีเลย์ที่ควบคุมกระบอกสูบ
const int SEED_SENSOR_PIN = 7;        // เซ็นเซอร์ตรวจจับเมล็ด (IR/Photoelectric)

// --- การตั้งค่าเวลาและ Timing (Timing Configuration) ---
const unsigned long SENSOR_DEBOUNCE_MS = 30;           // ป้องกันการสั่นของเซ็นเซอร์
const unsigned long CYLINDER_EXTEND_DURATION_MS = 500;  // เวลาที่กระบอกสูบยืด
const unsigned long CYLINDER_RETRACT_DURATION_MS = 250; // เวลาที่กระบอกสูบหด
const unsigned long ARM_TIMEOUT_MS = 2000;             // เวลาที่ระบบรอหลังจาก ARM
const unsigned long COMMAND_TIMEOUT_MS = 100;          // เวลาที่รอคำสั่งจาก Serial

// --- ตัวแปรสถานะระบบ (System State Variables) ---
bool isArmedForEjection = false;      // สถานะการเตรียมพร้อมยิง
bool systemReady = true;              // สถานะความพร้อมของระบบ
unsigned long lastSensorTriggerTime = 0;
unsigned long armCommandTime = 0;
unsigned long lastStatusTime = 0;

// --- ตัวแปรสถิติ (Statistics Variables) ---
unsigned long totalSeedsDetected = 0;
unsigned long goodSeedsCount = 0;
unsigned long badSeedsEjected = 0;
unsigned long missedEjections = 0;    // เมล็ดที่ควรยิงแต่พลาด

// --- ฟังก์ชันควบคุมกระบอกสูบ (Cylinder Control) ---
void actuateCylinder() {
  if (!systemReady) {
    Serial.println("ERROR: System not ready for ejection");
    return;
  }

  systemReady = false;
  Serial.println("EJECTING! Relay ON -> Cylinder Extend");
  
  // ยืดกระบอกสูบ
  digitalWrite(CYLINDER_RELAY_PIN, LOW);  // เปิดรีเลย์
  delay(CYLINDER_EXTEND_DURATION_MS);
  
  // หดกระบอกสูบ
  Serial.println("RETRACTING! Relay OFF -> Cylinder Retract");
  digitalWrite(CYLINDER_RELAY_PIN, HIGH); // ปิดรีเลย์
  delay(CYLINDER_RETRACT_DURATION_MS);
  
  badSeedsEjected++;
  systemReady = true;
  
  // ส่งสถานะกลับไป Python
  Serial.println("STATUS:BAD_SEED_EJECTED");
  Serial.print("STATS: Good=");
  Serial.print(goodSeedsCount);
  Serial.print(" Bad=");
  Serial.print(badSeedsEjected);
  Serial.print(" Total=");
  Serial.println(totalSeedsDetected);
}

// --- ฟังก์ชัน Watchdog สำหรับ ARM timeout ---
void checkArmTimeout() {
  if (isArmedForEjection && (millis() - armCommandTime > ARM_TIMEOUT_MS)) {
    Serial.println("WARNING: ARM timeout - system disarmed");
    isArmedForEjection = false;
    missedEjections++;
  }
}

// --- ฟังก์ชันแสดงสถานะระบบ ---
void printSystemStatus() {
  Serial.println("=== SYSTEM STATUS ===");
  Serial.print("Armed: ");
  Serial.println(isArmedForEjection ? "YES" : "NO");
  Serial.print("Ready: ");
  Serial.println(systemReady ? "YES" : "NO");
  Serial.print("Total Seeds: ");
  Serial.println(totalSeedsDetected);
  Serial.print("Good Seeds: ");
  Serial.println(goodSeedsCount);
  Serial.print("Bad Seeds Ejected: ");
  Serial.println(badSeedsEjected);
  Serial.print("Missed Ejections: ");
  Serial.println(missedEjections);
  Serial.println("====================");
}

// --- Setup Function ---
void setup() {
  // ตั้งค่า GPIO pins
  pinMode(CYLINDER_RELAY_PIN, OUTPUT);
  pinMode(SEED_SENSOR_PIN, INPUT_PULLUP);

  // เริ่มต้นสถานะ
  digitalWrite(CYLINDER_RELAY_PIN, HIGH);  // รีเลย์ปิด (กระบอกสูบหด)
  
  // เริ่มต้น Serial Communication
  Serial.begin(9600);
  delay(1000);
  
  // ข้อความเริ่มต้น
  Serial.println("========================================");
  Serial.println("Enhanced Coffee Sorter - Arduino Ready");
  Serial.println("Version: 2.0 | Arming Mode with Feedback");
  Serial.println("========================================");
  Serial.println("System is DISARMED and READY.");
  
  lastStatusTime = millis();
}

// --- Main Loop ---
void loop() {
  unsigned long currentTime = millis();
  
  // === ส่วนที่ 1: จัดการคำสั่งจาก Python/Serial ===
  if (Serial.available() > 0) {
    char command = Serial.read();
    
    switch (command) {
      case 'A':  // ARM command
        if (systemReady) {
          isArmedForEjection = true;
          armCommandTime = currentTime;
          Serial.println("System ARMED. Waiting for sensor trigger...");
        } else {
          Serial.println("ERROR: System not ready for ARM command");
        }
        break;
        
      default:
        Serial.print("Unknown command: ");
        Serial.println(command);
        break;
    }
  }
  
  // === ส่วนที่ 2: ตรวจสอบ ARM timeout ===
  checkArmTimeout();
  
  // === ส่วนที่ 3: ตรวจสอบเซ็นเซอร์และทำการคัดออก ===
  int sensorState = digitalRead(SEED_SENSOR_PIN);
  
  // ตรวจสอบเงื่อนไข: เซ็นเซอร์ทำงาน
  if (sensorState == HIGH && (currentTime - lastSensorTriggerTime > SENSOR_DEBOUNCE_MS)) {
    lastSensorTriggerTime = currentTime;
    totalSeedsDetected++;
    
    // ตรรกะการตัดสินใจ
    if (isArmedForEjection) {
      // นี่คือเมล็ดเสีย -> ยิงออก
      Serial.print("Bad seed detected by sensor (ID: ");
      Serial.print(totalSeedsDetected);
      Serial.print("). ");
      
      delay(1700);
      actuateCylinder();
      
      // ยกเลิกการเตรียมพร้อมทันที
      isArmedForEjection = false;
      Serial.println("System DISARMED automatically.");
      
    } else {
      // นี่คือเมล็ดดี -> ปล่อยผ่าน
      goodSeedsCount++;
      Serial.print("Good seed detected by sensor (ID: ");
      Serial.print(totalSeedsDetected);
      Serial.println("). Passing through.");
      
      // ส่งสถานะกลับไป Python
      Serial.println("STATUS:GOOD_SEED_PASSED");
    }
  }
  
  // === ส่วนที่ 4: ส่งสถานะระบบเป็นระยะ ===
  if (currentTime - lastStatusTime > 30000) {  // ทุก 30 วินาที
    Serial.print("Heartbeat - Armed: ");
    Serial.print(isArmedForEjection ? "YES" : "NO");
    Serial.print(" | Ready: ");
    Serial.print(systemReady ? "YES" : "NO");
    Serial.print(" | Total: ");
    Serial.println(totalSeedsDetected);
    lastStatusTime = currentTime;
  }
  
  // ดีเลย์เล็กน้อยเพื่อป้องกันการใช้ CPU สูง
  delay(10);
}