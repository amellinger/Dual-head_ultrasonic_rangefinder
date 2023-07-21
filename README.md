# Dual-head ultrasonic rangefinder

This repository contains the code and 3D printer files for a dual-head ultrasonic rangefinder, to be used in low-cost mechanical resonance experiments suitable for deployment in large lab courses with multiple stations. The motion of the two ends of a driven, damped spring oscillator is recorded with US-100 ultrasonic distance sensors and ESP8266 microcontrollers. Data is downloaded to a computer via Wi-Fi in a format suitable for analysis in Logger Pro. Sensor lag is compensated via a modified Savitzky-Golay filter. 

The built-in webserver uses [Chart.js](https://www.chartjs.org/) and [jQuery](https://jquery.com/).

For details, see W. Joysey and A. Mellinger, "Low-cost ultrasonic distance measurement in a mechanical resonance experiment", https://arxiv.org/abs/1906.08778.

## Installation
- Download and install the [Arduino IDE](https://www.arduino.cc/en/main/software). Use the legacy (1.8.xx) IDE, as SPIFFS file system upload is not supported in the Arduino IDE 2.
- From the IDE, install the [ESP8266 board](https://arduino-esp8266.readthedocs.io/en/latest/installing.html). The recommended version is 2.4.2. For versions 2.6.0 and later, the macro ``ESP8266_NEW_SOFTWARESERIAL`` should be defined in the code to account for synax changes in ``SoftwareSerial``. Not yet fully tested.
- Select the proper board (e.g. LOLIN(WEMOS) D1 R2 & mini)
- Install the [Arduino ESP8266 filesystem uploader](https://github.com/esp8266/arduino-esp8266fs-plugin).
- Close the serial monitor. Select the "Tools > ESP8266 Sketch Data Upload" menu item. This should start uploading the files into ESP8266 flash file system.
- Compile and upload the code.
