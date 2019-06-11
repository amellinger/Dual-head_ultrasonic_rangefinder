/* Distance data logger with ESP8266 internal storage 
 * Axel Mellinger
 * Central Michigan University
 * 
 * V 1.0   2019-03-26
 * V 1.1   2019-05-06
 *         If cmich_open network not found, ESP8266 will switch to ad-hoc(soft AP) WiFi
 * V 1.2   2019-05-31
 *         Added phase shift compensation to Savitzky-Golay filter
 * V 1.3   2019-06-11
 *         Code clean-up
 *   
 */

#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <FS.h>   //Include File System Headers
#include <time.h>
#include <Adafruit_SSD1306.h>
#include <SoftwareSerial.h>;

#define NPTS 2000         // max. number of data points
#define MAXDIST 1000      // max. distance in mm
#define MAXWIN 17        // max. SG filter windwow size
#define WiFiMAXTRY 40     // max. WiFi connection attempts before switching to soft AP mode

#define TDELAY 10.08e-3   // Time shift between US-100 sensor readings

#define SCREEN_WIDTH 128  // OLED display width, in pixels
#define SCREEN_HEIGHT 64  // OLED display height, in pixels

#define SSID_AP_BASE "PHY17x_"

// Declaration for an SSD1306 display connected to I2C (SDA, SCL pins)
#define OLED_RESET LED_BUILTIN // Reset pin # (or -1 if sharing Arduino reset pin)
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);


//WiFi Connection configuration
const char* ssid = "cmich_open";
char ssid_ap[20];
//const char* password = "secret";


String s="";
byte mac[6];          



short int dist_mm1[NPTS], dist_mm2[NPTS];  // data arrays
short int NewPoint[NPTS]; // Filtered data 
int dt[NPTS];
float h0[MAXWIN][MAXWIN];
float h1[MAXWIN][MAXWIN];
float h2[MAXWIN][MAXWIN];

#define US100_TX1  14 // D5
#define US100_RX1  12 // D6
#define US100_TX2  15 // D8
#define US100_RX2  13 // D7
 
SoftwareSerial US100_sensor_1(US100_RX1, US100_TX1, false, 255);
SoftwareSerial US100_sensor_2(US100_RX2, US100_TX2, false, 255);


ESP8266WebServer server(80);


// convert (and invert) distance to meters
// top sensor
float dist1(float dist) {
  double tmp = sqrt((dist*dist/1.0e8) - ((.025/2)*(.025/2)));
  if (isnan(tmp)) {
    return 0;
  } else {
    return 0.18-tmp;
  }
}

// bottom sensor
float dist2(float dist) {
  double tmp = sqrt((dist*dist/1.0e8) - ((.025/2)*(.025/2)));
  if (isnan(tmp)) {
    return 0;
  } else {
    return tmp;
  }
}





// Need to define our own 'mystream', since server.streamFile cannot be called
// multiple times
void mystream(String filepath) {
  #define bufsize 512
  char buf[bufsize+1];
  int cnt;
  Serial.println(filepath);
  File file = SPIFFS.open(filepath.c_str(), "r");
  int fa = file.available();
  while (fa>0) {
    if (fa>bufsize) {
      cnt = bufsize;
    }
    else {
      cnt = fa;
    }
    Serial.println(fa);
    Serial.println(cnt);
    file.readBytes(buf, cnt);
    buf[cnt] = 0x0;
    server.sendContent(buf);
    fa = file.available();
  }
  file.close();
}



void clearDisplayLines() {
  display.setCursor(0, 42);
  display.println(F("                                  "));  
  display.display();
  display.setCursor(0, 56);
  display.println(F("                     "));
  display.display();
  display.setCursor(0, 56);
}


void outputChart(int n) {
  s = F("<canvas id=\"line-chart\" width=\"700\" height=\"350\"></canvas>\n");
  s += F("<script>\nnew Chart(document.getElementById(\"line-chart\"), {\ntype: 'scatter',\n");
  s += F("data: { \n\
          datasets: [{ \n\
              label: 'Top', \n\
              showLine: true, \n\
              fill: false, \
              backgroundColor: \"#0000AA\", \n\
              borderColor: \"#0000AA\", \n\
              borderWidth: 1, \n\
              pointRadius: 1, \n\
              pointBorderWidth: 0,\n\
              data: [");
  server.sendContent(s);

  // data 1
  for (int i=0; i<n; i++) {
    s = "{x:"+ String(dt[i]/1000.0, 4) + ",y:" + String(dist1(dist_mm1[i]), 4) + "}";
    if (i<n-1) {
      s += ",";
    }
    s += "\n";
    //\t" + String(distance2, 4) + "\r\n";
    server.sendContent(s);
  }
  
  s = F("              ] \n\},\n{");
  s +=F("     \
              label: 'Bottom', \n\
              showLine: true, \n\
              fill: false, \
              backgroundColor: \"#00AA00\", \n\
              borderColor: \"#00AA00\", \n\
              borderWidth: 1, \n\
              pointRadius: 1, \n\
              pointBorderWidth: 0,\n\
              data: [");
  server.sendContent(s);

  // data 2
  for (int i=0; i<n; i++) {
    s = "{x:"+ String(dt[i]/1000.0, 4) + ",y:" + String(dist2(dist_mm2[i]), 4) + "}";
    if (i<n-1) {
      s += ",";
    }
    s += "\n";
    //\t" + String(distance2, 4) + "\r\n";
    server.sendContent(s);
  }
  s = F("              ] \n\}\n");
  s +=F("]},\n");
  server.sendContent(s);

  // labels etc.
  s = F("\
    options: { \n\
        responsive: false, \n\
        scales: { \n\
            xAxes: [{ \n\
                type: 'linear', \n\
                position: 'bottom', \n\
                scaleLabel: { \n\
                    display: true, \n\
                    labelString: 'Time (s)' \n\
                } \n\
            }], \n\
            yAxes: [{ \n\
                type: 'linear', \n\
                position: 'left', \n\
                scaleLabel: { \n\
                    display: true, \n\
                    labelString: 'Position (m)' \n\
                } \n\
            }] \n\
        } \n\
    }\n");
  s += F("});\n</script>\n"); 
  server.sendContent(s);     
}


bool handleFileRead(String path) { // send the right file to the client (if it exists)
  String path1, path2;
  Serial.println("handleFileRead: " + path);
  int n;
  int w;
  unsigned long t0;
  unsigned int DistArray[2];
  String clientinfo = "RmtIP " + server.client().remoteIP().toString();
  Serial.println(clientinfo);
  clearDisplayLines;
  display.setCursor(0, 42);
  display.println(clientinfo);
  display.display();

//  delay(500);


  if (path.endsWith("/")) {
    server.setContentLength(CONTENT_LENGTH_UNKNOWN);
    server.send (200, "text/html", "");
    path1 = path + "index_header.html";
    path2 = path + "index_footer.html";
    if (SPIFFS.exists(path1)) {                           // If the file exists
      mystream(path1);
    }

    // Read number of points
    if (server.hasArg("numPoints")) {
      n = server.arg("numPoints").toInt();
      Serial.print("n = ");
      Serial.println(n);
    } else {
      n = 600;
    }

     // Read filtering Window
    if (server.hasArg("filterWindow")) {
      w = server.arg("filterWindow").toInt();
      Serial.print("w = ");
      Serial.println(w);
    } else {
      w = 7;
    }
    // write form
    s = F("<form action=\"/\" method=\"post\">\n");
    s += "Number of points (max. " + String(NPTS,DEC) +"): <input type=\"number\" id=\"npts\" name=\"numPoints\" min=\"10\" max=\"" + String(NPTS,DEC) + "\" value=\"" + String(n,DEC) + "\"><br>\n";
    s += "Filtering Window (max. " + String(MAXWIN,DEC) +"): <input type=\"number\" id=\"npts\" name=\"filterWindow\" min=\"1\" max=\"" + String(MAXWIN,DEC) +  "\" value=\"" + String(w,DEC) + "\"><br>\n";
    s += F("<input type=\"submit\" id=\"run\" value=\"Acquire Data\" formmethod=\"post\" name=\"acquire\"/>\n");
    s += F("<input type=\"button\" id=\"download\" value=\"Save Data\" /><p></p>\n");
    s += F("</form>\n");
    s += F("<div style=\"width: 100%; display: table;\">\n<div style=\"display: table-row\">\n<div style=\"width: 320px; display: table-cell; vertical-align:top;\">\n"); 
    s += F("<textarea readonly id=\"content\" rows=\"40\" cols=\"35\">");
    server.sendContent(s);



    
   
    // If "Acquire Data" button was pressed: collect and output data
    if (server.hasArg("acquire")) {
      /*
      // Dummy output
      for (int i=0; i<2000; i++) {
        String str=String(i,DEC)+"\t"+String(random(1000)/1000., 3)+"\t"+String(random(1000)/1000., 2)+"\n";
        server.sendContent(str);
      }
      */

      // begin measurement loop
      clearDisplayLines();
      display.setCursor(0, 56);     // Start at top-left corner
      display.println("Acquiring data...");
      display.display();
    
      digitalWrite(LED_BUILTIN, LOW);
      t0 = millis();
      if (n>NPTS) { n=NPTS; }       // prevent buffer overflow
      if (n<10) { n=10; }
      for (int i=0; i<n; i++) {
        // read distance in units of 0.1 mm to decrease quantization error
        if (read_US100_Dist_sequential(&US100_sensor_1, &US100_sensor_2, DistArray)) {
          i--;                      // repeat measurement if data is invalid
        } 
        else {
          // record time
          dt[i] = millis()-t0;
          dist_mm1[i] = 10*DistArray[0];
          dist_mm2[i] = 10*DistArray[1];
        }
      }
      // end measurement loop

      digitalWrite(LED_BUILTIN, HIGH);
      display.setCursor(0, 56);
      display.println("                    ");
      display.setCursor(0, 56);
      display.println("Downloading data...");
      display.display();



      // Savitzky-Golay Filtering starts here

      
      double tstep = 1.0*dt[n-1]/n/1000.;             // time step in seconds
      SG_Filter(dist_mm1, n, w, 0);
      SG_Filter(dist_mm2, n, w, -TDELAY/tstep);
//      SG_Filter(dist_mm2, n, m, 0.0);

      // S-G filtering ends here


      // output data
      s = F("Vernier Format 2\r\n");
      s += F("Motion Sensor Distance Readings\r\n");
      s += F("Data Set\r\n");
      s += F("Time\tTop\tBottom\r\n");
      s += F("T\tTop\tBottom\r\n");
      s += F("s\tm\tm\r\n");
      server.sendContent(s);
      
      for (int i=0; i<n; i++) {
        s = String(dt[i]/1000.0, 4) + "\t" + String(dist1(dist_mm1[i]), 4) + "\t" + String(dist2(dist_mm2[i]), 4) + "\r\n";
        server.sendContent(s);
      }

    }
   
    s = F("</textarea>\n</div>\n<div style=\"display: table-cell;\">\n");
    server.sendContent(s);

    if (server.hasArg("acquire")) {
      outputChart(n);  
    }

    s = F("</div>\n</div>\n</div>\n");
    s += F("</body>\n</html>");
    server.sendContent(s);


    clearDisplayLines();
    return true;
  }
  
  if (path.endsWith(".js")) {
    Serial.println(F("Javascript file requested"));
    if (SPIFFS.exists(path)) {
      File dataFile = SPIFFS.open(path.c_str(), "r");
      if (server.streamFile(dataFile, "application/javascript") != dataFile.size()) {
        clearDisplayLines();
        return false;
      }
    } else {
      clearDisplayLines();
      return false;
    }
    clearDisplayLines();
    return true;
  }

  if (path.endsWith("robots.txt")) {
    Serial.println(F("robots.txt file requested"));
    if (SPIFFS.exists(path)) {
      File dataFile = SPIFFS.open(path.c_str(), "r");
      if (server.streamFile(dataFile, "text/plain") != dataFile.size()) {
        clearDisplayLines();
        return false;
      }
    } else {
      clearDisplayLines();
      return false;
    }
    clearDisplayLines();
    return true;
  }
  
  Serial.println("\tFile Not Found");
  clearDisplayLines();
  return false;                                         // If the file doesn't exist, return false
}


// The next three functions are from P. A. Gorry, Analyt. Chem. 62, 570 (1990).
// Weights() calculates the Savitzky-Golay filter coefficients for the j-th data point for the t-th Least-Square point of the s-th derivative.
// Filter window width is 2m+1 points; polynomial order is n.
double Weights(int j, int t, int m, int n, int s){
  int k; 
  double sum; 
  sum = 0;
  //Serial.printf("%i %i %i %i %i\n", j, t, m, n, s);
  for (k = 0; k<=n; k++){
    sum += (2.0*k+1)* (GenFact(2*m, k )/GenFact(2.0*m+k+1,k+1)) *GramPoly(j,m,k,0)*GramPoly(t,m,k,s);
  }
  return sum;
}

double GenFact(int a, int b){
  int j;
  double gf;
  gf = 1;
  for(j= (a-b+1); j<=a; j++){
    gf *= j;
  }
  return gf;
}
double GramPoly(int i, int m, int k, int s){
  if(k>0){
    return (4.0*k-2)/(k*(2*m -k+1))* (i* GramPoly(i,m,k-1,s) + s*GramPoly(i,m,k-1,s-1))-((k-1)*(2.0*m+k))/(k*(2.0*m-k+1))*GramPoly(i,m,k-2,s);
      
  }
  else if((k==0) && (s==0)){
    return 1;
  }
  else return 0;
}

   


// Savitzky-Golay filter with phase shift correction
void SG_Filter(short int *data, int n, int w, double shift) {
  double sum0, sum1, sum2;
  int m = (w-1)/2;
 
  // Pre-calculate the S-G weight factors
  for (int i = -m; i <= m; i++){
    for (int t = -m; t <= m; t++){
       h0[i+m][t+m] = Weights(i,t,m,2,0);
       h1[i+m][t+m] = Weights(i,t,m,2,1);
       h2[i+m][t+m] = Weights(i,t,m,2,2);
    }
  }   
  
  // Convolution
  for (int i=0; i<n; i++) {
    int t;
    sum0 = 0.0; sum1 = 0.0; sum2 = 0.0;
    for (int j= -m; j <= m ; j++){  
      if ((m <= i) && ( i <= (n-1)-m)) {
         t=0;                                 // sufficient number (m) points to the left and right
      }   
      else if (i < m) {
         t = i - m;                           // start of data set (points missing to the left)
      }   
      else {
        t = i-(n-1-m);                        // end of data set (points missing to the right) 
      }
      sum0 += h0[j+m][t+m]*data[i+j-t];       // smoothed value at point i
      sum1 += h1[j+m][t+m]*data[i+j-t];       // first derivative at point i  
      sum2 += h2[j+m][t+m]*data[i+j-t];       // second derivative at point i
    }
    NewPoint[i] = round(sum0 + sum1*shift + 0.5 * sum2 * shift*shift);   // phase shift correction
    yield(); 
  }
      

  

  Serial.printf("In function: first + last data points: %i  %i   %i %i\n", data[0], data[1], data[n-2], data[n-1]);
  Serial.printf("In function: first + last filtered data points: %i %i   %i %i\n", NewPoint[0], NewPoint[1], NewPoint[n-2], NewPoint[n-1]);
  for(int i=0; i < n; i++){
      data[i] = NewPoint[i];
  }
}





byte read_US100_Dist_sequential(SoftwareSerial* SerPort1, SoftwareSerial* SerPort2, unsigned int* DistArray) {
  unsigned int MSByteDist = 0;
  unsigned int LSByteDist = 0;
  unsigned int mmDist1 = 0;
  unsigned int mmDist2 = 0;
  byte error = 0;

  
    
  SerPort1->flush(); // clear the serial buffer
  SerPort1->write(0x55);           // start distance measurement (sensor 1)
  delay(9);                        // wait for data to be returned
  float tmp;
//  Serial.printf("data availability: %i  -  %i\n", SerPort1->available(), SerPort2->available());
  if (SerPort1->available()) {     // verify that 2 bytes are available for sensor 1
    tmp=micros();
    MSByteDist = SerPort1->read(); // read out US-100 data
    LSByteDist  = SerPort1->read();
    mmDist1  = MSByteDist * 256 + LSByteDist; // distance in mm
  }
  else {
    mmDist1 = 13;
    error = 1;
  }

  

  SerPort2->flush();
  SerPort2->write(0x55);           // start for distance measurement (sensor 2) 
  delay(9);
  if (SerPort2->available()) {     // verify that 2 bytes are available for sensor 2
    MSByteDist = SerPort2->read(); // read out US-100 data
    LSByteDist  = SerPort2->read();
    mmDist2  = MSByteDist * 256 + LSByteDist; // distance in mm
  }
  else {
    mmDist2 = 13;
    error = 1;
  }
  
  DistArray[0] = mmDist1;
  DistArray[1] = mmDist2;
  if ((mmDist1>MAXDIST) || (mmDist2>MAXDIST)) { error = 1; }
  if (error) { Serial.println("Distance measurement error!"); }
//  Serial.printf("mmDist1: %i DistArray[0] : %04x \n" , mmDist1, DistArray[0]);
//  Serial.printf("mmDist2: %i DistArray[1] : %04x \n\n" , mmDist2, DistArray[1]);

  return error;
}



void setup() {
  int rnd;

  Serial.begin(115200);
  US100_sensor_1.begin(9600);
  US100_sensor_2.begin(9600);

  delay(500);

  // Initialize OLED
  // SSD1306_SWITCHCAPVCC = generate display voltage from 3.3V internally
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) { // Address 0x3D for 128x64
    Serial.println(F("SSD1306 allocation failed"));
    for(;;); // Don't proceed, loop forever
  }
  // Show welcome message and IP address on OLED display
  display.clearDisplay();
  display.setTextColor(WHITE, BLACK); // Draw white text on black background
  display.setCursor(0, 0);     // Start at top-left corner
  display.cp437(true);         // Use full 256 char 'Code Page 437' font
  display.setRotation(2);

  digitalWrite(LED_BUILTIN, HIGH);  // turn off LED

  //Initialize File System
  SPIFFS.begin();
  Serial.println(F("File System Initialized"));

  // Print the MAC address
  WiFi.macAddress(mac);
 
  //Connect to wifi Network
  WiFi.setPhyMode(WIFI_PHY_MODE_11N);
  WiFi.setOutputPower(50) ;         // limits output power
  WiFi.mode(WIFI_STA);              // station mode
//  WiFi.begin(ssid, password);     //Connect to your WiFi router
  WiFi.begin(ssid);
  
  Serial.println("");

  display.setCursor(0, 42);
  display.println(F("Connecting to"));
  display.print(ssid);
  display.display();

  
  // Wait for connection
  unsigned int cnt = 0;
  while ((WiFi.status() != WL_CONNECTED) && (cnt<WiFiMAXTRY)) {
    delay(500);
    Serial.print(".");
    cnt++;
  }
  clearDisplayLines();

  String IP;
  
// Switch to Ad-hoc mode if WiFi connection was unsuccessful
  if (cnt>=WiFiMAXTRY) {
 
    sprintf(ssid_ap, "%s%02x%02x%02x", SSID_AP_BASE, mac[3], mac[4], mac[5]);
  
    WiFi.mode(WIFI_AP);
    WiFi.softAP(ssid_ap);
    IP = WiFi.softAPIP().toString();
    //Show SSID and IP address in serial monitor
    Serial.println("");
    Serial.print(F("Access point SSID: "));
    Serial.println(ssid_ap);
    Serial.print(F("Use this URL to connect: "));
    Serial.println(IP);  //IP address assigned to your ESP
  }
  else {
    IP = WiFi.localIP().toString();
    //If connection successful show IP address in serial monitor
    Serial.println("");
    Serial.print(F("Connected to "));
    Serial.println(ssid);
    // Print the IP address
    Serial.print(F("Use this URL to connect: "));
    Serial.print("http://");
    Serial.print(WiFi.localIP());
    Serial.println("/");
  }

  // Print MAC address 
  s = "MAC address: " + String(mac[0], HEX);
  for (int i=1; i<=5; i++) {
    s += ":"+String(mac[i], HEX);
  }
  Serial.println(s);

 

//  Initialize Webserver
  server.onNotFound([]() {                              // If the client requests any URI
    if (!handleFileRead(server.uri()))                  // send it if it exists
      server.send(404, "text/plain", "404: Not Found"); // otherwise, respond with a 404 (Not Found) error
  });
  server.begin();  
  Serial.println(F("Server started"));




  // Display welcome message
  display.setCursor(0, 0);     // Start at top-left corner
  display.setTextSize(2);      // Normal 1:1 pixel scale
  display.println(F("Welcome to\n   CMU\n Physics!"));
  display.display();
  delay(5000);
  display.clearDisplay();

  display.setTextSize(1);      // Normal 1:1 pixel scale
  display.setCursor(0, 0);     // Start at top-left corner


  if (cnt>=WiFiMAXTRY) {
    display.print(F("SSID: "));
    display.println(ssid_ap);
    display.println(F("Point your browser to\nIP Address"));
    display.println("");
    display.print("http://");
    display.println(IP);
  }
  else {
    display.print(F("SSID: "));
    display.println(ssid);
    display.println(F("Point your browser to\nIP Address"));
    display.println("");
    display.print("http://");
    display.println(WiFi.localIP());
  }
  display.display();

  // Read sensors once (first data point after power-up appears to be bad)
  unsigned int DistArray[2];
  read_US100_Dist_sequential(&US100_sensor_1, &US100_sensor_2, DistArray);

  // Configure GPIO poins
  pinMode(LED_BUILTIN, OUTPUT); 

}


void loop() {
 server.handleClient();
// Serial.println("In Loop");
 delayMicroseconds(100); // 100 µs needed to keep WiFi active. Why? (ping fails after a few seconds w/o this statement)
// delay(0);  
}
