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
 * V 1.4   2019-07-23
 *         Code clean-up   
 * V 1.5   2020-07-22         
 *         Changed ssid from cmich_open to CMICH-DEVICE
 *         ESP8266_NEW_SOFTWARESERIAL switch to update SoftwareSerial call syntax to work with BoardManager 2.7.2 (old syntax needs 2.4.2 or earlier)
 * V 1.6   2020-10-13
 *         Omitted http:// in IP display on OLED screen. For long numbers, the last digit was sometimes wrapped
 *         to the next line.
 *         Reduced "Welcome to CMU Physics" delay from 5000 to 4000 ms.
 *         
 * V 1.7   2020-11-03       
 *         Device connects to www.cmich.edu after web server initialization. Makes the device ping-able
 *         on CMICH-DEVICE if they get a 141.209 IP address
 *         (disabled by default)
 *         
 * V 1.8   2023-07-21
 *         Updated jquery to 3.7.0 (security fix). Increased chart line width ("borderWidth") from 1 to 2.
 *         Added CMU logo. Set filtering min. window size to 3.
 *            
 * V 1.9   2023-07-24
 *         Update website appearance with CMU logo, CSS and a loading spinner. Explanatory text for number of points and filter window.
 *         
 * V 1.10  2023-07-25
 *         Fixed extraneous </td> and missing </tr> elements, and other HTML/CSS errors.
 *         
 * V 1.11  2023-08-17
 *         Cleared remote RmtIP on display after data acquisition.
 *         
 * V 2.0   2024-01-15
 *         Streams data via serial output. Web interface still works.
 *         OLED display polishing
 */

#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <FS.h>   //Include File System Headers
#include <time.h>
#include <Adafruit_SSD1306.h>
#include <SoftwareSerial.h>

// Switch that changes the SoftwareSerial syntac. Activate for esp8266 board versions 2.6.0 or higher.
// Note: There are still some serial communication problems between the ESP8266 and the US-100. Thus,
//       it is recommended to disable ESP8266_NEW_SOFTWARESERIAL and use esp8266 board version 2.4.2
// #define ESP8266_NEW_SOFTWARESERIAL

#define NPTS 1500         // max. number of data points
#define MAXDIST 1000      // max. distance in mm
#define MINWIN 3        // min. SG filter windwow size
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
const char* ssid = "CMICH-DEVICE";
//const char* ssid = "Arubatest";
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

#ifdef ESP8266_NEW_SOFTWARESERIAL
  SoftwareSerial US100_sensor_1;
  SoftwareSerial US100_sensor_2;
#else
  SoftwareSerial US100_sensor_1(US100_RX1, US100_TX1, false, 255);
  SoftwareSerial US100_sensor_2(US100_RX2, US100_TX2, false, 255);
#endif





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
    Serial.print(F("file available = ")); Serial.println(fa);
    Serial.print(F("cnt = ")); Serial.println(cnt);
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
              borderWidth: 2, \n\
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
  
  s = F("              ] \n},\n{");
  s +=F("     \
              label: 'Bottom', \n\
              showLine: true, \n\
              fill: false, \
              backgroundColor: \"#00AA00\", \n\
              borderColor: \"#00AA00\", \n\
              borderWidth: 2, \n\
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
  s = F("              ] \n}\n");
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
      n = 400;
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
    s += F("<table>\n");
    s += F("  <tr>\n");
    s +=   "     <td>Number of points (max. " + String(NPTS,DEC) +"): <br><span style=\"font-size:9pt;\">Acquiring a larger number of data points takes more time.</span></td> <td style=\"text-align: right;\"><input type=\"number\" id=\"npts\" name=\"numPoints\" style=\"width: 4em; text-align: right;\" min=\"10\" max=\"" + String(NPTS,DEC) + "\" value=\"" + String(n,DEC) + "\"></td>\n";
    s += F("     <td rowspan=\"2\" style=\"background-color: #ffffff; padding: 0px 50px\"> <input type=\"submit\" id=\"run\" value=\"Acquire Data\" formmethod=\"post\" name=\"acquire\"> ");
    s += F("     <td rowspan=\"2\" style=\"background-color: #ffffff; padding: 0px 20px\"> <input type=\"button\" id=\"download\" value=\"Save Data\"></td>\n");
    s += F("  </tr>\n   <tr>\n");
    s += "     <td>Filter window size (" + String(MINWIN,DEC) + "..." + String(MAXWIN,DEC) +"): <br><span style=\"font-size:9pt;\">Large filter window sizes result in smoother curves.</span></td> <td style=\"text-align: right;\"><input type=\"number\" id=\"filtWin\" name=\"filterWindow\" style=\"width: 4em; text-align: right;\" min=\"" + String(MINWIN,DEC) + "\" max=\"" + String(MAXWIN,DEC) +  "\" value=\"" + String(w,DEC) + "\"></td>\n";
    s += F("  </tr>\n");
    s += F("</table>\n<p></p>\n");
    s += F("</form>\n");
    s += F("<div style=\"width: 100%; display: table;\">\n<div style=\"display: table-row\">\n<div style=\"width: 320px; display: table-cell; vertical-align:top;\">\n"); 
    s += F("<textarea readonly id=\"content\" rows=\"30\" cols=\"31\">");
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

///////////////////////
      acquire_data(n, w);
///////////////////////
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
    Serial.println(F("Data downloaded"));
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

  if (path.endsWith(".png")) {
   Serial.println(F("image file requested"));
    if (SPIFFS.exists(path)) {
      File dataFile = SPIFFS.open(path.c_str(), "r");  
      if (server.streamFile(dataFile, "image/png") != dataFile.size()) {
        return false;
      }
    } else {
      clearDisplayLines();
      return false;
    }
    clearDisplayLines();
    return false;
  }
  
  Serial.println("\tFile Not Found");
  clearDisplayLines();
  return false;                                         // If the file doesn't exist, return false
}


void acquire_data(int n, int w){
  unsigned long t0;
  unsigned int DistArray[2];
 
  // begin measurement loop
  clearDisplayLines();
  display.setCursor(0, 56);
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
  Serial.println(F("Start data download"));
  s = F("Vernier Format 2\r\n");
  s += F("Motion Sensor Distance Readings\r\n");
  s += F("Data Set\r\n");
  s += F("Time\tTop\tBottom\r\n");
  s += F("t\td_top\td_bottom\r\n");
  s += F("s\tm\tm\r\n");
  server.sendContent(s);

  Serial.println("# Data Begin");
  for (int i=0; i<n; i++) {
    s = String(dt[i]/1000.0, 4) + "\t" + String(dist1(dist_mm1[i]), 4) + "\t" + String(dist2(dist_mm2[i]), 4) + "\r\n";
    server.sendContent(s);
    Serial.print(s); // Serial data output
  }
  Serial.println("# Data End");

  display.setCursor(0, 56);
  display.println("                    ");
  display.display();

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
  #ifdef ESP8266_NEW_SOFTWARESERIAL
    US100_sensor_1.begin(9600, SWSERIAL_8N1, US100_RX1, US100_TX1, false, 255);
    US100_sensor_2.begin(9600, SWSERIAL_8N1, US100_RX2, US100_TX2, false, 255);
  #else 
    US100_sensor_1.begin(9600);
    US100_sensor_2.begin(9600);
  #endif

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
    Serial.print("http://");//
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
  delay(1000);

  display.clearDisplay();
  display.setTextSize(1);      // Normal 1:1 pixel scale
  display.setCursor(0, 0);     // Start at top-left corner


/*
  // Web connectivity check:
  // Connect to test webserver (initiate connection from device)
  const char* host = "www.cmich.edu";
  // Use WiFiClient class to create TCP connections
  WiFiClient client;
  const int httpPort = 80;  if (!client.connect(host, httpPort)) {
    Serial.println("Web connectivity check failed");
    display.println(F("Web connectivity\ncheck: FAILED"));
    display.display();
    delay(2000);
    // return;
  } else {
    Serial.print(host);
    Serial.println(" connected");
    display.println(F("Web connectivity\ncheck: ok"));
    display.display();
    delay(2000);
  }
*/

  display.clearDisplay();
  display.setCursor(0, 0);     // Start at top-left corner



//  // We now create a URI for the request
//  String url = "/";
//
//  Serial.print("Requesting URL: ");
//  Serial.println(url);
//
//  // This will send the request to the server
//  client.print(String("GET ") + url + " HTTP/1.1\r\n" +
//               "Host: " + host + "\r\n" +
//               "Connection: close\r\n\r\n");
//  unsigned long timeout = millis();
//  while (client.available() == 0) {
//    if (millis() - timeout > 5000) {
//      Serial.println(">>> Client Timeout !");
//      client.stop();
//      return;
//    }
//  }
//
//  // Read all the lines of the reply from server and print them to Serial
//  while (client.available()) {
//    String line = client.readStringUntil('\r');
//    Serial.print(line);
//  }

  Serial.println();
  Serial.println("####################closing connection");





  if (cnt>=WiFiMAXTRY) {
    display.print(F("SSID: "));
    display.println(ssid_ap);
    display.println(F("Point your browser to\nIP Address"));
    display.println("");
//    /display.print("http://");
    display.println(IP);
  }
  else {
    display.print(F("SSID: "));
    display.println(ssid);
    display.println(F("Point your browser to\nIP Address"));
    display.println("");
//    display.print("http://")/;
    display.println(WiFi.localIP());
  }
  display.display();

  // Read sensors once (first data point after power-up appears to be bad)
  unsigned int DistArray[2];
  read_US100_Dist_sequential(&US100_sensor_1, &US100_sensor_2, DistArray);

  // Configure GPIO poins
  pinMode(LED_BUILTIN, OUTPUT); 

}

int incomingByte = 0; // for incoming serial data
int n = 400;
int w = 7;
int cnt = 0;


void loop() {
  server.handleClient();
// Serial.println("In Loop");
  delayMicroseconds(100); // 100 µs needed to keep WiFi active. Why? (ping fails after a few seconds w/o this statement)
  //display.clearDisplay();
//  display.setCursor(0, 10);     // Start at top-left corner
//  display.println("Start...");
//  display.display();

  if (Serial.available() > 0) {
    // read the incoming byte:
    incomingByte = Serial.read();
    
    // say what you got:
    Serial.print("I received: ");
    Serial.println(incomingByte, DEC);
    if (incomingByte==58){      // colon character intiates data acquisition
      n = Serial.parseInt();
      w = Serial.parseInt();

      if ((n>10) && (n<NPTS) && (w>=MINWIN) && (w<=MAXWIN)) {
        acquire_data(n, w);
      }
      Serial.println("Data output done.");      
    }
  }
// delay(0);  
}
