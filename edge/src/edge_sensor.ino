#define VIBRATION_PIN   2      // Must be pin 2 or 3 (interrupt pins)
#define WINDOW_MS       500    // Count pulses over 500ms window
#define DEBOUNCE_MS      20    // Ignore bounces within 20ms
#define THRESHOLD_LOW     1    // 1+ pulses  → LOW
#define THRESHOLD_MEDIUM  4    // 4+ pulses  → MEDIUM
#define THRESHOLD_HIGH   10    // 10+ pulses → HIGH

volatile int  pulseCount   = 0;
volatile unsigned long lastPulseTime = 0;
unsigned long windowStart  = 0;

void onVibration() {
  unsigned long now = millis();
  if (now - lastPulseTime > DEBOUNCE_MS) {
    pulseCount++;
    lastPulseTime = now;
  }
}

String classifyIntensity(int count) {
  if (count >= THRESHOLD_HIGH)   return "HIGH";
  if (count >= THRESHOLD_MEDIUM) return "MEDIUM";
  if (count >= THRESHOLD_LOW)    return "LOW";
  return "NONE";
}

void setup() {
  Serial.begin(9600);
  pinMode(VIBRATION_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(VIBRATION_PIN), onVibration, RISING);
  windowStart = millis();
  Serial.println("{\"device\":\"SW420_UNO\",\"event\":\"READY\"}");
}

void loop() {
  if (millis() - windowStart >= WINDOW_MS) {
    noInterrupts();
    int count  = pulseCount;
    pulseCount = 0;
    interrupts();
    windowStart = millis();

    String intensity = classifyIntensity(count);

    if (intensity != "NONE") {
      Serial.print("{\"device\":\"SW420_UNO\",\"event\":\"VIBRATION\",");
      Serial.print("\"intensity\":\""); Serial.print(intensity);
      Serial.print("\",\"pulse_count\":"); Serial.print(count);
      Serial.print(",\"timestamp_ms\":"); Serial.print(millis());
      Serial.println("}");
    }

    static unsigned long lastHeartbeat = 0;
    if (millis() - lastHeartbeat >= 5000) {
      lastHeartbeat = millis();
      Serial.print("{\"device\":\"SW420_UNO\",\"event\":\"HEARTBEAT\",");
      Serial.print("\"timestamp_ms\":"); Serial.print(millis());
      Serial.println("}");
    }
  }
}