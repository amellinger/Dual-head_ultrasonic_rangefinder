# Dual-head ultrasonic rangefinder

This repository contains the code and 3D printer files for a dual-head ultrasonic rangefinder, to be used in low-cost mechanical resonance experiments suitable for deployment in large lab courses with multiple stations. The motion of the two ends of a driven, damped spring oscillator is recorded with US-100 ultrasonic distance sensors and ESP8266 microcontrollers. Data is downloaded to a computer via Wi-Fi in a format suitable for analysis in Logger Pro. Sensor lag is compensated via a modified Savitzky-Golay filter. 

The built-in webserver uses [Chart.js](https://www.chartjs.org/) and [jQuery](https://jquery.com/).

For details, see W. Joysey and A. Mellinger, "Low-cost ultrasonic distance measurement in a mechanical resonance experiment", https://arxiv.org/abs/1906.08778.
