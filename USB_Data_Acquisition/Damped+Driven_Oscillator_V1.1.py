# -*- coding: utf-8 -*-
"""
Created on Thu Aug. 4, 2023

USB readout of the ESP8266-based ultrasonic sensor

2023-11-26: V 1.0
2024-03-19: V 1.0   Bug-fix for "Save Data" button

@author: melli1a
"""


WindowTitle = 'Damped + Driven Oscillator  V 1.1'


import numpy as np
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi']=75

from matplotlib.widgets import SpanSelector
import os
from uncertainties import ufloat
import traceback

from scipy.optimize import curve_fit      # Curve fitting module
from fit_tools import err_corr_matrix     # Convert covariance matrix to uncertainties and correlation coefficients

import sys
if getattr(sys, 'frozen', False):
    import pyi_splash


import PySimpleGUI as sg
import serial
import serial.tools.list_ports

global  fitline1, fitline2, fitline3, fitline4, arrow1, arrow2, arrow3, arrow4, dash1, dash2, span, span2,\
    text1, text2, text3, text4, textdamped, textexp
arrow1=arrow2=arrow3=arrow4=dash1=dash2=span=span2=text1=text2=text3=text4=textdamped=textexp=None


# fit limits
global tmin, tmax
tmin=tmax=None

global data, serial_reply
data = np.zeros((0,2))
global ser
global ser_devices
ser_devices = []


# serial stream data markers
DATA_BEGIN = '# Data Begin'
DATA_END = '# Data End'

MIN_SIZE = (1256, 645) # window minimum size

NMax = 600 # max. number of data points


#%% Status update
def update_status(status, error=False):
    window['-STATUS-'].update(value=status)
    window.refresh()    
    if error:
        tr = traceback.format_exc()
        print('type traceback:', type(tr))
        print(tr)
        print('string(tr):', str(tr))
        s = str('Show this error message to your instructor.\n\n' + str(status) + '\n' + str(traceback.format_exc()))
        sg.popup_error(s, title='An error occured')

#%%
def detect_serial_ports():
    global ser_devices
    ports = serial.tools.list_ports.comports()
    ser_devices = [p.device for p in ports]
    print(ser_devices)
    if len(ser_devices)>0:
        window['-SerPort-'].update(values=ser_devices, value=ser_devices[-1])

def connect_serial_port():
    global ser
    port = values['-SerPort-']
    print('port:', port)
    if window['-SerConnect-'].get_text()=='Connect':                    # connect serial port
        if port != '':
            try:
                ser = serial.Serial(port, baudrate=115200, timeout=15)  # open serial port
                print(ser.name)                                         # check which port was really used
            except serial.SerialException:
                window['-SerConnect-'].update('Connect')
                window['-START-'].update(disabled=True)
                update_status('No sensor connected.')
            else:
                window['-SerConnect-'].update('Disconnect')
                window['-START-'].update(disabled=False)
                update_status('Ready.')
        else:
            try:
                ser.close()
                window['-START-'].update(disabled=True)
                window['-SerConnect-'].update('Connect')
                update_status('No sensor connected.')
            except Exception as error:
                update_status('No sensor connected.')

    else:                                                               # disconnect serial port
        window['-SerConnect-'].update('Connect')
        ser.close()
        window['-START-'].update(disabled=True)
        

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller 
    From https://stackoverflow.com/questions/7674790/bundling-data-files-with-pyinstaller-onefile
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# Default values
N0 = 400
FilterWindow0 =7
#serial_port = '/dev/ttyUSB0'

global event, values, inputData

#sg.theme('DarkBlue3')
sg.theme('DarkPurple3')
sg.SetOptions(background_color='#6A0032', text_element_background_color='#6A0032', element_background_color='#6A0032',
               input_text_color='#FFFFFF', input_elements_background_color='#556B6F')

# All the stuff inside your window.
layout = [  [sg.Frame(layout=[[sg.Image(resource_path('CMU-PHY_Logo_268px.png'), expand_x=True, expand_y=True ), 
                               sg.Text(text='Damped + Driven Oscillator', font=('Arial 36 bold'), expand_x=True, text_color='#FFC82E',
                                       justification='left')],
            ],
                      title='', size=(1226,90))],
            [sg.Frame(layout=[[sg.Column([[sg.Canvas(key='controls_cv')], [sg.Canvas(key='-CANVAS-', size=(520,None))]]),
                               sg.Column([[sg.Frame(layout=[[sg.Text('Serial Port:')],
                                                 [sg.Combo(ser_devices, default_value='', size=(15, None), key='-SerPort-', readonly=True, pad=(5, (0,20)))],
                                                 [sg.Button('Detect Serial Port', size=14, key='-DetectSerialPort-')],
                                                 [sg.Button('Connect', size=14, key='-SerConnect-')]],
                                         title='Device', title_color='#FFC82E', size=(150,200)),
                                           sg.Frame(layout=[[sg.Column([[sg.Text(f'# of points\n(max. {NMax})')]], size=(110,40)),
                                                             sg.Column([[sg.Input(N0, size=(6, None), justification='right', key='-NPts-')]],
                                                                       vertical_alignment='top')],
                                                            [sg.Column([[sg.Text('Acquiring a larger number\nof points takes more time.\n', font=('', 8))]])],
                                                            [sg.Column([[sg.Text('Filter Window')]], size=(110, 25)),
                                                             sg.Column([[sg.Input(FilterWindow0, size=(6, None), justification='right', 
                                                                                  key='-FilterWindow-' )]], 
                                                                       vertical_alignment='top')],
                                                            [sg.Column([[sg.Text('Larger filter window sizes\nresult in smoother curves.', font=('', 8))]])]                                                        
                                                           ], 
                                                    title="Acqusition Parameters", title_color='#FFC82E', size=(190,200))],
                                          [sg.Frame(layout=[[sg.Button('Acquire Data', key='-START-', size=12, disabled=True),
                                                             sg.Button('Fit Driven Osc.', key='-FitDriven-', size=12),
                                                             sg.Button('Close', size=12)],
                                                            [sg.Input(key='-SaveData-', enable_events=True, visible=False, size=10), sg.FileSaveAs(button_text='Save Data', file_types=(("TXT Files", "*.txt"),("ALL Files", "*.*")),
                                                                           size=12, key='-SaveData_Button-' ),
                                                             sg.Button('Fit Damped Osc.', key='-FitDamped-', size=12)]
                                                            ], 
                                                    title='Actions', title_color='#FFC82E', size=(350,96))],
                                          [sg.Frame(layout=[
                                                            [sg.Text('', size=(15,1), key='-T00-'),
                                                             sg.Text('', size=(4,1), key='-T01-'),
                                                             sg.Text('', key='-T02-', text_color='#ffffff')],
                                                            [sg.Text('', size=(15,1), key='-T10-'),
                                                             sg.Text('', size=(4,1), key='-T11-'),
                                                             sg.Text('', key='-T12-', text_color='#ffffff')],
                                                            [sg.Text('', size=(15,1), key='-T20-'),
                                                             sg.Text('', size=(4,1), key='-T21-'),
                                                             sg.Text('', key='-T22-', text_color='#ffffff')],
                                                            [sg.Text('', size=(15,1), key='-T30-'),
                                                             sg.Text('', size=(4,1), key='-T31-'),
                                                             sg.Text('', key='-T32-', text_color='#ffffff')],
                                                            [sg.Text('', size=(15,1), key='-T40-'),
                                                             sg.Text('', size=(4,1), key='-T41-'),
                                                             sg.Text('', key='-T42-', text_color='#ffffff')]
                                                            ],
                                                    title='Analysis', title_color='#FFC82E', size=(350,170))]
                                          ])]], title='')],
            [sg.Frame(layout=[[sg.Text('Status: ', font=('', 9)), sg.Text('', key='-STATUS-', font=('', 9))]], title='', size=(1226,29), vertical_alignment='top')]
        ]

# Analysis text labels
OutText = [[['Ang. Driving Freq.:', 'ωd  ='],
            ['Top Amplitude:',      'A0  ='],
            ['Bottom Amplitude:',   'A   ='],
            ['Phase Shift: ',       'Δϕ  ='],
            ['',                    '']],
           [['Initial Amplitude:',  'Ai  ='],
            ['Damping Param.:',  '𝛾   ='],
            ['Angular Frequency:',  'ω   ='],
            ['Phase:',              'ϕ   ='],
            ['Zero Position:',      'x0  =']]]


#%% Draw plot with toolbar:
# https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_Matplotlib_Embedded_Toolbar.py
    

def draw_figure_w_toolbar(canvas, fig, canvas_toolbar):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    if canvas_toolbar.children:
        for child in canvas_toolbar.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    figure_canvas_agg.draw()
    toolbar = Toolbar(figure_canvas_agg, canvas_toolbar)
    toolbar.update()
    figure_canvas_agg.get_tk_widget().pack(side='right', fill='both', expand=1)
    return figure_canvas_agg

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

class Toolbar(NavigationToolbar2Tk):
    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)
        
#%%
def onselect(t1, t2):
    global span, span2, tmin, tmax
    if t1!=t2:
        tmin = t1
        tmax = t2
    else:
        # set artists coordinates to zero to allow autoscaling if data range is reduced
        t1 = t2 = 0
        for ar in list(span2.artists):
            if type(ar)==matplotlib.patches.Rectangle:
                ar.set_xy([0,0])
            if type(ar)==matplotlib.lines.Line2D:
                ar.set_xdata([0,0])
        if len(data)>0:
            tmin = data[0,0]
            tmax = data[-1,0]

    if len(data)>0:
        indmin, indmax = np.searchsorted(data[:,0], (tmin, tmax))
    else:
        indmin = indmax = 0
    print(f'indmin = {indmin}      indmax = {indmax}         tmin = {tmin:.4f}      tmax = {tmax:.4f}')
    # draw same rectangle 
    if span is not None:
        span.remove()
    span = ax.axes.axvspan(t1, t2, alpha=0.2, facecolor='tab:gray')
    fig_canvas_agg.draw()

#    indmax = min(len(x) - 1, indmax)

#    region_x = x[indmin:indmax]
#    region_y = y[indmin:indmax]



#%%
def validate_int(s, lower, upper):
    try:
        value = int(s)
    except ValueError:
        update_status(f'Error: Invalid input {s}.', error=True)
        return np.NaN
    else:
        if (value>=lower) and (value<=upper):
            return value
        else:
            update_status(f'Error: Out of range input {value}. Should be within {lower}...{upper}.', error=True)
            return np.NaN

#%%

def xdr(t, A, omega, phi, x0):
    return A*np.cos(omega*t+phi) + x0

# Fit driven oscillator waveforms
def fit_xdr(t, x):
    p0 = np.zeros(4)
    print(f'fit_xdr:  x= {x}')
    p0[0] = 0.5*(np.max(x)-np.min(x))
    
    # find omega with FFT
    dt = (t[-1]-t[0])/(len(x)-1)
    f = np.fft.rfftfreq(len(x), dt)
    print('f = ', f)
    fft = np.fft.rfft(x)
    print('fft = ', fft)
    fft[0] = 0
    idx = np.abs(fft).argmax()
    fmax = f[idx]
    p0[1] = 2*np.pi*fmax
    p0[2] = np.angle(fft[idx])
    if p0[2]<0:
        p0[2] += 2*np.pi
    p0[3] = 0.5*(np.max(x)+np.min(x))
    print(f'p0 = {p0}')
    popt, pcov = curve_fit(xdr, t, x, p0=p0, method='trf', bounds=((0, 0, -0.5*np.pi, -np.inf), (np.inf, np.inf, 2.5*np.pi, np.inf)))
    print(f'popt = {popt}\n\n')
    perr, pcorr = err_corr_matrix(pcov)
    return popt, perr

def fit_driven():
    global data, tt, xx1, xx2, fitline1, fitline2, arrow1, arrow2, arrow3, arrow4, dash1, dash2, tmin, tmax, \
           text1, text2, text3, text4

    if (tmin!=None) and (tmax!=None):
        indmin, indmax = np.searchsorted(data[:,0], (tmin, tmax))    
    else:
        indmin, indmax = (0,0)
    print(f'start of fit_driven: indmin: {indmin},    indmax: {indmax}     tmin: {tmin}  tmax:{tmax}')
    if (len(data[:,0])>4) and (indmax-indmin>4):
        clear_fit()   # remove old labels

        try:
            top_popt, top_perr = fit_xdr(data[indmin:indmax,0], data[indmin:indmax,1])
            print(top_popt, top_perr)
            bottom_popt, bottom_perr = fit_xdr(data[indmin:indmax,0], data[indmin:indmax,2])
            print(bottom_popt, bottom_perr)
            
            A0 = ufloat(top_popt[0], top_perr[0])
            A = ufloat(bottom_popt[0], bottom_perr[0])
            phi1 = ufloat(top_popt[2], top_perr[2])
            phi2 = ufloat(bottom_popt[2], bottom_perr[2])
            phi = phi1-phi2
            # phi = top_popt[2] - bottom_popt[2]
            while phi.n<0:
                phi += 2*np.pi
            while phi.n>2*np.pi:
                phi -= 2*np.pi
    
            omega_d1 = ufloat(top_popt[1], top_perr[1])
            omega_d2 = ufloat(bottom_popt[1], bottom_perr[1])
            omega_d = (omega_d1+omega_d2)/2.0
            
            # Draw fit lines
            tt = np.linspace(data[0,0], data[-1,0], 1000, dtype=np.float64)
            xx1 = xdr(tt, top_popt[0], top_popt[1], top_popt[2], top_popt[3])
            xx2 = xdr(tt, bottom_popt[0], bottom_popt[1], bottom_popt[2], bottom_popt[3])
            fitline1.set_data(tt, xx1)
            # x = data[:,1]
            # fitline1.set_data(tt, xdr(tt, 0.5*(np.max(x)-np.min(x)), 3.0, -1.5, 0.5*(np.max(x)+np.min(x))))
            fitline2.set_data(tt, xx2)
            if not(fitline1 in list(ax.get_lines())):
                ax.add_line(fitline1)
            if not(fitline2 in list(ax2.get_lines())):
                ax2.add_line(fitline2)
    
            # Draw amplitude arrows
            t1per = min(data[-1,0], 1.1*2*np.pi/top_popt[1])
    
            tt = tt[tt<=t1per] # Trim data to 1 period
            xx1 = xdr(tt, top_popt[0], top_popt[1], top_popt[2], top_popt[3])
            xx2 = xdr(tt, bottom_popt[0], bottom_popt[1], bottom_popt[2], bottom_popt[3])
            
            
            
            idx1 = np.argmax(xx1)
            idx2 = np.argmax(xx2)
            add_t = 0
            if idx2<idx1:
                add_t = 2*np.pi/bottom_popt[1]
            # print('index:', idx1, idx2)
            # mark period T
            arrow1 = mpatches.FancyArrowPatch((tt[idx1], top_popt[3]+1.1*A0.n), (tt[idx1]+2*np.pi/omega_d.n, top_popt[3]+1.1*A0.n), mutation_scale=20, color='#cf7400', zorder=4)        # mark amplitude A0
            ax.add_patch(arrow1)
            # mark amplitude A0
            arrow2 = mpatches.FancyArrowPatch((tt[idx1], top_popt[3]), (tt[idx1], xx1[idx1]), mutation_scale=20, color='#bf0000', zorder=4)
            ax.add_patch(arrow2)
            # mark amplitude A
            arrow3 = mpatches.FancyArrowPatch((tt[idx2]+add_t, bottom_popt[3]), (tt[idx2]+add_t, xx2[idx2]), mutation_scale=20, color='#bf0000', zorder=4)
            ax2.add_patch(arrow3)
            # mark phase difference phi
            arrow4 = mpatches.FancyArrowPatch((tt[idx1], bottom_popt[3]), (tt[idx1]+phi.n/omega_d.n, bottom_popt[3]), mutation_scale=20, color='#00afaf', zorder=4)
            ax2.add_patch(arrow4)
            
            # Vertical dashed lines 
            dash1 = ax.axvline(tt[idx1], 0, 0.5, color='k', ls='dashed', lw=1)
            dash2 = ax2.axvline(tt[idx1], 0.5, 1.0, color='k', ls='dashed', lw=1)
            
            # Annotate arrows
            
            text1 = ax.text(tt[idx1]+np.pi/omega_d.n, top_popt[3]+1.18*A0.n, "$T = 2\pi/\omega_\mathsf{d}$", fontsize=16, color='#cf7400', zorder=5, va='bottom', ha='center')
            text2 = ax.text(tt[idx1], top_popt[3] + 0.2*(xx1[idx1]-top_popt[3]), " $A_0$", fontsize=16, color='#bf0000', zorder=5)
            text3 = ax2.text(tt[idx2]+add_t, bottom_popt[3] + 0.2*(xx2[idx2]-bottom_popt[3]), " $A$", fontsize=16, color='#bf0000', zorder=5)
            text4 = ax2.text(tt[idx1], bottom_popt[3] - 0.1*(xx2[idx2]-bottom_popt[3]), "$\Delta\phi/\omega_d$", fontsize=16, color='#00afaf', ha='left', va='top', zorder=5)
        
            # Text output of fit results
            for i in range(len(OutText[0])):
                for j in range(len(OutText[0][i])):
                    key = f'-T{i}{j}-'
                    print(i,j, key)
                    window[key].update(value=OutText[0][i][j])
            
            window['-T02-'].update(value=f'({omega_d.n:.3f} ± {omega_d.s:.3f}) rad/s')
            window['-T12-'].update(value=f'({A0.n:.4f} ± {A0.s:.4f}) m')
            window['-T22-'].update(value=f'({A.n:.4f} ± {A.s:.4f}) m')
            window['-T32-'].update(value=f'({phi.n:.2f} ± {phi.s:.2f}) rad')
            window['-T42-'].update(value='')
            window.refresh()
    
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        except Exception as error:
            update_status(error, True)

    print(f'*****end of fit_driven: indmin: {indmin},    indmax: {indmax}     tmin: {tmin}  tmax:{tmax}')


#%%


def xdmp(t, A0, gamma, omega, phi, x0):
    return A0 * np.exp(-gamma*t) * np.cos(omega*t+phi) + x0

# Fit driven oscillator waveforms
def fit_xdmp(t, x):
    p0 = np.zeros(5)
    print(f'fit_xdmp:  x= {x}')
    p0[0] = 0.5*(np.max(x)-np.min(x))
    
    # find omega with FFT
    dt = (t[-1]-t[0])/(len(x)-1)
    f = np.fft.rfftfreq(len(x), dt)
    print('f = ', f)
    fft = np.fft.rfft(x)
    print('fft = ', fft)
    fft[0] = 0
    idx = np.abs(fft).argmax()
    fmax = f[idx]
    p0[1] = 2.0
    p0[2] = 2*np.pi*fmax
    p0[3] = np.angle(fft[idx])
    if p0[2]<0:
        p0[2] += 2*np.pi
    p0[4] = 0.5*(np.max(x)+np.min(x))
    print(f'p0 = {p0}')
    popt, pcov = curve_fit(xdmp, t, x, p0=p0, method='trf', 
                           bounds=((0, 0, 0, -np.inf, -np.inf), (np.inf, np.inf, np.inf, 2*np.pi, np.inf)))
    print(f'popt = {popt}\n\n')
    perr, pcorr = err_corr_matrix(pcov)
    return popt, perr

def fit_damped():
    global data, tt, xx1, xx2, fitline1, fitline2, fitline3, fitline4, arrow1, arrow2, arrow3, arrow4, dash1, dash2, tmin, tmax, \
           text1, text2, text3, text4, textexp, textdamped    
    if (tmin!=None) and (tmax!=None):
        indmin, indmax = np.searchsorted(data[:,0], (tmin, tmax))    
    else:
        indmin, indmax = (0,0)
    print(f'start of fit_damped: indmin: {indmin},    indmax: {indmax}     tmin: {tmin}  tmax:{tmax}')
    if (len(data[:,0])>4) and (indmax-indmin>4):
        clear_fit()

        try:
            bottom_popt, bottom_perr = fit_xdmp(data[indmin:indmax,0]-data[indmin,0], data[indmin:indmax,2])
            print(bottom_popt, bottom_perr)
            
            A0 = ufloat(bottom_popt[0], bottom_perr[0])
            gamma = ufloat(bottom_popt[1], bottom_perr[1])
            omega = ufloat(bottom_popt[2], bottom_perr[2])
            phi = ufloat(bottom_popt[3], bottom_perr[3])
            while phi.n<0:
                phi += 2*np.pi
            while phi.n>2*np.pi:
                phi -= 2*np.pi
            x0 = ufloat(bottom_popt[4], bottom_perr[4])
        
            # Draw fit lines
            tt = np.linspace(data[indmin,0], data[-1,0], 1000, dtype=np.float64)-data[indmin,0]
            xx = xdmp(tt, bottom_popt[0], bottom_popt[1], bottom_popt[2], bottom_popt[3], bottom_popt[4])
            fitline2.set_data(tt+data[indmin,0], xx)
            if not(fitline2 in list(ax2.get_lines())):
                ax2.add_line(fitline2)
                
            # envelopes
            fitline3.set_data(tt+data[indmin,0], bottom_popt[0]*np.exp(-bottom_popt[1]*tt)+bottom_popt[4])
            if not(fitline3 in list(ax2.get_lines())):
                ax2.add_line(fitline3)
            fitline4.set_data(tt+data[indmin,0], -bottom_popt[0]*np.exp(-bottom_popt[1]*tt)+bottom_popt[4])
            if not(fitline4 in list(ax2.get_lines())):
                ax2.add_line(fitline4)
        
            arrowpos1_x = tt[0]+data[indmin,0] + 1.1/gamma.n
            arrowpos1_y = bottom_popt[0]*np.exp(-bottom_popt[1]*(arrowpos1_x-data[indmin,0]))+bottom_popt[4]
            textpos1_x = arrowpos1_x + 0.4/gamma.n
            textpos1_y = arrowpos1_y+0.20*A0.n
        
            arrowpos2_x = tt[0]+data[indmin,0] + 0.55*2.0*np.pi/omega.n
            arrowpos2_y = xdmp((arrowpos2_x-data[indmin,0]), A0.n, gamma.n, omega.n, phi.n, x0.n)
            textpos2_x = arrowpos2_x + 0.7*2.0*np.pi/omega.n
            textpos2_y = arrowpos2_y-0.02*A0.n
            
            print(f'********** fit_damped: tt[0]={tt[0]}',  arrowpos2_x, arrowpos2_y, textpos2_x, textpos2_y)
    
            # Text output of fit functions
            textexp = ax2.annotate('$A_\mathsf{i}\, e^{-\gamma t} + x_0$', xy=(arrowpos1_x, arrowpos1_y), xytext=(textpos1_x, textpos1_y),
                arrowprops=dict(facecolor='#707000', shrink=0.05, color='#707000', width=1, headwidth=7, headlength=10),
                color='#707000', fontsize=14, zorder=5
                )
            textdamped = ax2.annotate('$A_\mathsf{i}\, e^{-\gamma t}\, \cos(\omega t+\phi)+x_0$', xy=(arrowpos2_x, arrowpos2_y), 
                xytext=(textpos2_x, textpos2_y),
                arrowprops=dict(facecolor='#303030', shrink=0.01, color='#303030', width=1, headwidth=7, headlength=10),
                color='#303030', fontsize=14, zorder=5, va='center'
                )
        
            # Text output of fit results
            for i in range(len(OutText[1])):
                for j in range(len(OutText[1][i])):
                    key = f'-T{i}{j}-'
                    print(i,j, key)
                    window[key].update(value=OutText[1][i][j])
            
            window['-T02-'].update(value=f'({A0.n:.4f} ± {A0.s:.4f}) m')
            window['-T12-'].update(value=f'({gamma.n:.4f} ± {gamma.s:.4f}) 1/s')
            window['-T22-'].update(value=f'({omega.n:.3f} ± {omega.s:.3f}) rad/s')
            window['-T32-'].update(value=f'({phi.n:.2f} ± {phi.s:.2f}) rad')
            window['-T42-'].update(value=f'({x0.n:.3f} ± {phi.s:.3f}) m')
            window.refresh()
        except Exception as error:
            update_status(error, True)
         
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

#%%
def clear_fit():
    global fitline1, fitline2, arrow1, arrow2, arrow3, arrow4, dash1, dash2, tmin, tmax, \
           text1, text2, text3, text4, textdamped, textexp


    # Remove text labels from previous fits
    if text1 in list(ax.texts):
        text1.remove()
    if text2 in list(ax.texts):
        text2.remove()
    if text3 in list(ax2.texts):
        text3.remove()
    if text4 in list(ax2.texts):
        text4.remove()
    if textdamped in list(ax2.texts):
        textdamped.remove()
    if textexp in list(ax2.texts):
        textexp.remove()

    # hide fits, arrows and dashed lines from previous run
    if fitline1 in list(ax.get_lines()): 
        fitline1.remove()
    if fitline2 in list(ax2.get_lines()): 
        fitline2.remove()
    if fitline3 in list(ax2.get_lines()): 
        fitline3.remove()
    if fitline4 in list(ax2.get_lines()): 
        fitline4.remove()
    if arrow1 in list(ax.patches): 
        arrow1.remove()
    if arrow2 in list((ax.patches)): 
        arrow2.remove()
    if arrow3 in list(ax2.patches): 
        arrow3.remove()
    if arrow4 in list(ax2.patches): 
        arrow4.remove()
    if dash1 in list(ax.get_lines()): 
        dash1.remove()
    if dash2 in list(ax2.get_lines()): 
        dash2.remove()    
#%%

def do_measurement():
    global event, values, inputData, data, serial_reply, amp, arrow1, arrow2, arrow3, arrow4, tmin, tmax
    update_status('acquiring data...')
    n = validate_int(values['-NPts-'], 10, NMax)
    print('n=', n)
    w = validate_int(values['-FilterWindow-'], 3, 15)
    
    if not np.isnan(n) and not np.isnan(w):
        try:
            ser.write(b':')            # ':' is the trigger string to start data acquisition by the ESP8266
            s = f'{n}\n{w}'
            ser.write(s.encode())
        except serial.SerialException:
            update_status('Serial error. Check USB connection.', error=True)
            fig_canvas_agg.draw()
            window['-START-'].update(disabled=True)
            window['-SerConnect-'].update('Connect')
            return False
        
        print("Reading serial data...")
        try:
            serial_reply = ser.read_until(DATA_END.encode('ascii')) # read_until needs a byte object, not a string
        except serial.SerialException:
            update_status('Serial error. Check USB connection.', error=True)
            fig_canvas_agg.draw()
            window['-START-'].update(disabled=True)
            window['-SerConnect-'].update('Connect')
            return False
        print("Done reading serial data.")
        tmp = str(serial_reply)
        # print(tmp)
        try:
            tmp2 = tmp.split(DATA_BEGIN)[1].split(DATA_END)[0]
            data_l = tmp2.split('\\r\\n')
            data_l2 = []
            for l in data_l:
                if l!='':
                    data_l2.append(l.split('\\t'))
    
            data = np.array(data_l2, dtype=np.float32)

########## Test code #######    
            
            # amp = 0.1*np.random.rand() + 0.02
            # r1 = 2*np.pi*np.random.rand()
            # r2 = r1+np.pi*np.random.rand()
            # wd = 1.5+8*np.random.rand()
            # # wd = 3.5
            
            # data = np.zeros((n,3))
            # data[:,0] = np.linspace(0, 5, n)
            # data[:,1] = (0.4*(np.random.rand(n)-0.5)+1.0) * amp*np.sin(wd*data[:,0]-r1) + 1.1*amp
            # data[:,2] = (0.4*(np.random.rand(n)-0.5)+1.0) * 3*amp*np.sin(wd*data[:,0]-r2) + 0.8*amp
            
            # if np.random.randint(2):
            #     data[:,1] = 0.05
            #     data[:,2] = np.where(data[:,0]>0.5, 
            #                           (0.4*(np.random.rand(n)-0.5)+1.0) * 0.08*np.exp(-(data[:,0]-0.5)*1.0) * np.cos(8*(data[:,0]-0.5)) + 0.06,
            #                           (0.06+0.08)*(0.07*(np.random.rand(n)-0.5)+1.0))

########## Test code #######    
    
            t = data[:,0]
            top = data[:,1]
            bottom = data[:,2]
    
            line1.set_data(t, top)
            line2.set_data(t, bottom)
            
            clear_fit()
    
            # hide value output text
            window['-T02-'].update(value='')
            window['-T12-'].update(value='')
            window['-T22-'].update(value='')
            window['-T32-'].update(value='')
            window['-T42-'].update(value='')
            window.refresh()
            
            ax.set_autoscale_on(True)
            ax2.set_autoscale_on(True)        
    
            ax.relim()
            ax.autoscale_view()         # automatic axis scaling
            ax2.relim()
            ax2.autoscale_view()         # automatic axis scaling
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
    
            # set vertical axes to same scale 
            ax1lim = ax.get_ylim()
            ax2lim = ax2.get_ylim()
            av1 = (ax1lim[1]+ax1lim[0])/2.0   # mid-point 
            av2 = (ax2lim[1]+ax2lim[0])/2.0
            d1 = ax1lim[1]-ax1lim[0]
            d2 = ax2lim[1]-ax2lim[0]
            if d1>d2:
                ax2.set_ylim([av2-d1/2.0, av2+d1/2.0])
            else:
                ax.set_ylim([av1-d2/2.0, av1+d2/2.0])
                
            # Set fit limits
            if span2!=None:
                if span2.extents[1]-span2.extents[0] != 0:
                    tmin, tmax = span2.extents
                else:
                    tmin = data[0,0]
                    tmax = data[-1,0]
         
            
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            # a=3/0   # raise error
            update_status('Ready.')

        except ValueError as error:
            print("Value error.", error)
            connect_serial_port()
            update_status(f' Value error. {error}', error=True)
        except NameError as error:
            print("Name error.", error)
            update_status(f' Name error. {error}', error=True)
        except IndexError as error:
            print("Index error.", error)
            connect_serial_port()
            update_status(f' Index error. {error}', error=True)
        except Exception as error:
            print("Error:", error)
            update_status(error, True)
        # except:
        #     print("Unexpected error:", sys.exc_info()[0])
        #     window['-STATUS-'].update(value=f' Unexpected error. {sys.exc_info()[0]}')
        #     raise

    window.refresh()
    fig_canvas_agg.draw()

    


def save_data(filename):
    global data
    f = open(filename, mode='w')
    f.write('Vernier Format 2\nMotion Sensor Distance Readings\nData Set\nTime	Top	Bottom\nt\td_top\td_bottom\ns	m	m\n')
    for i in range(len(data)):
        f.write(f'{data[i,0]:.4f}\t{data[i,1]:.4f}\t{data[i,2]:.4f}\n')
    f.close()



#################################################################################
#%% Close splash screen
if getattr(sys, 'frozen', False):
    pyi_splash.close()

# Create the Window
window = sg.Window(WindowTitle, layout, font='Arial 10', finalize=True,
                   enable_close_attempted_event=True, resizable=True)
events, values = window.read(timeout=10)  # timeout is in ms

# enforce minimum window size
width, height = tuple(map(int, window.TKroot.geometry().split("+")[0].split("x")))
# print(width, height)
window.TKroot.minsize(max(MIN_SIZE[0], width), max(MIN_SIZE[1], height))

N = int(values['-NPts-'])
FilterWindow = int(values['-FilterWindow-'])


inputData=np.empty((2, N))
inputData[:] = np.NAN

# initialize serial port
detect_serial_ports()
events, values = window.read(timeout=10)  # timeout is in ms
print('values = ', values)
print("Connecting serial port...")
connect_serial_port()


        
#%%  Initialize plot 

#fig = plt.figure(0)
fig = Figure(figsize=(11.1, 5.8), tight_layout=True, facecolor='#dddddd')  # using plt.figure() opens second plot window
fig.clf()
ax = fig.add_subplot(211)
ax.set_facecolor('#eeeeee')
ax.axes.tick_params(direction='in', which='both', top=True, right=True)
#ax.set_xlabel('Time (s)')
ax.set_ylabel(' Top Distance (m)')

ax2 = fig.add_subplot(212)
ax2.sharex(ax)
ax2.set_facecolor('#eeeeee')
ax2.axes.tick_params(direction='in', which='both', top=True, right=True)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Bottom Distance (m)')

ax.set_axisbelow(True)   # Grid lines below other elements
ax2.set_axisbelow(True)
ax.grid(which='both', axis='both')
ax2.grid(which='both', axis='both')


line1, = ax.plot(np.NAN, np.NAN, 'o-', color='b', label='top', markersize=4, alpha=0.4)
line2, = ax2.plot(np.NAN, np.NAN, 's-', color='g', label='bottom', markersize=4, alpha=0.4)
fitline1, = ax.plot(np.NAN, np.NAN, '-', color='k', lw=1.4)
fitline2, = ax2.plot(np.NAN, np.NAN, '-', color='k', lw=1.4)
fitline3, = ax2.plot(np.NAN, np.NAN, '--', color='#707000', lw=1.0)
fitline4, = ax2.plot(np.NAN, np.NAN, '--', color='#707000', lw=1.0)
ax.autoscale(True, axis='y')
ax2.autoscale(True, axis='y')


fit_text = ax2.text(0.98, 0.02, 'Click and drag in the bottom window to set the fit range. Click to unset range.', fontsize=10, 
                    alpha=0.6, transform=ax2.transAxes, ha='right', color='#202020', zorder=5)

ax.figure.canvas.draw()
ax2.figure.canvas.draw()
fig.canvas.draw()
plt.show()

# add the plot to the main window
fig_canvas_agg = draw_figure_w_toolbar(window['-CANVAS-'].TKCanvas, fig, window['controls_cv'].TKCanvas)
# fig_canvas_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)

span2 = SpanSelector(
    ax2,
    onselect,
    "horizontal",
    useblit=False,
    props=dict(alpha=0.2, facecolor="tab:gray"),
    interactive=True,
    drag_from_anywhere=True
)

# try:
#     a=3/0
# except Exception as e:
#     update_status('Division by zero.', True)
update_status('Ready.')
window.refresh()        




# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
#    event, values = window.read(timeout=1000)
    print('Event: ', event)
    if event == sg.WIN_CLOSE_ATTEMPTED_EVENT or event == 'Close': # if user closes window or clicks cancel
        break
    if event == '-START-':
        do_measurement()
    if event == '-SaveData-':
        save_data(values['-SaveData-'])
    if event == '-DetectSerialPort-':
        detect_serial_ports()
    if event == '-SerConnect-':
        connect_serial_port()
    if event == '-FitDriven-':
        fit_driven()
    if event == '-FitDamped-':
        fit_damped()
        
    print('You entered ', values)
    fig_canvas_agg.draw()
    
try:    
    ser.close()    
except:
    pass
window.close()