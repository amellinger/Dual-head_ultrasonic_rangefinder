pyinstaller -F -p . --windowed --splash Ultrasonic_USB_Data_Acquisition_splash.png -i Damped+Driven_Oscillator.ico --add-data "CMU-PHY_Logo_268px.png:." --hidden-import='PIL._tkinter_finder' Damped+Driven_Oscillator_V1.0.py fit_tools.py