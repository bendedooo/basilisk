#
#  ISC License
#
#  Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
#
#  Permission to use, copy, modify, and/or distribute this software for any
#  purpose with or without fee is hereby granted, provided that the above
#  copyright notice and this permission notice appear in all copies.
#
#  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
#  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
#  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
#  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
#  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#
r"""
Overview
--------

``OpNavScenarios/plotting/OpNav_Plotting.py`` contains the plotting routines. None of the files are saved,
but are shown when the scenario is run with python. Saving is left to the user's discretion.

"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# import scipy.optimize
from Basilisk.utilities import macros as mc
from Basilisk.utilities import simHelpers
from matplotlib.patches import Ellipse

color_x = 'dodgerblue'
color_y = 'salmon'
color_z = 'lightgreen'
m2km = 1.0 / 1000.0
ns2min = 1/60.*1E-9

mpl.rcParams.update({'font.size' : 8 })
# If a specific style for plotting wants to be used
try:
    plt.style.use("myStyle")
    params = {'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 8, 'xtick.labelsize': 7,
              'ytick.labelsize': 7, 'text.usetex': True}
    mpl.rcParams.update(params)
except:
    pass

# def fit_sin(tt, yy):
#     """
#     Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"
#     """
#     tt = np.array(tt)
#     yy = np.array(yy)
#     ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
#     Fyy = abs(np.fft.fft(yy))
#     guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
#     guess_amp = np.std(yy) * 2.**0.5
#     guess_offset = np.mean(yy)
#     guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])
#
#     def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
#     popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
#     A, w, p, c = popt
#     f = w/(2.*np.pi)
#     fitfunc = lambda t: A * np.sin(w*t + p) + c
#     return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

def show_all_plots():
    plt.show()

def clear_all_plots():
    plt.close("all")

def omegaTrack(rError, covar):
    colorsInt = len(mpl.pyplot.get_cmap("inferno").colors)/(10.)
    colorList = []
    for i in range(10):
        colorList.append(mpl.pyplot.get_cmap("inferno").colors[int(i*colorsInt)])

    t = rError[:, 0] * ns2min

    plt.figure(num=2210101, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    plt.plot(t , rError[:, 1]*180./np.pi, color = colorList[2] , label= 'Error')
    plt.plot(t, 3*np.sqrt(covar[:, 0,0])*180./np.pi, color=colorList[8], linestyle = '--', label=r'Covar (3-$\sigma$)')
    plt.plot(t, -3*np.sqrt(covar[:, 0,0])*180./np.pi, color=colorList[8], linestyle = '--')
    plt.legend(loc='best')
    plt.ylabel(r'$\mathbf{\omega}_{' + str(2) +r'}$ Error in $\mathcal{C}$ ($^\circ$/s)')
    plt.ylim([-0.07,0.07])
    plt.xlabel('Time (min)')
    # plt.savefig('RateCam1.pdf')

    plt.figure(num=2210201, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    plt.plot(t , rError[:, 2]*180./np.pi, color = colorList[2])
    plt.plot(t, 3*np.sqrt(covar[:, 1,1])*180./np.pi, color=colorList[8], linestyle = '--')
    plt.plot(t, -3*np.sqrt(covar[:, 1,1])*180./np.pi, color=colorList[8], linestyle = '--')
    plt.ylabel(r'$\mathbf{\omega}_{' + str(3) +r'}$ Error in $\mathcal{C}$ ($^\circ$/s)')
    plt.ylim([-0.07,0.07])
    plt.xlabel('Time (min)')
    # plt.savefig('RateCam2.pdf')



def vecTrack(ref, track, covar):

    colorsInt = len(mpl.pyplot.get_cmap("inferno").colors)/(10.)
    colorList = []
    for i in range(10):
        colorList.append(mpl.pyplot.get_cmap("inferno").colors[int(i*colorsInt)])

    rError = np.copy(ref)
    rError[:,1:] -= track[:,1:]
    errorDeg = np.zeros([len(rError[:, 0]), 2])
    covDeg = np.zeros([len(rError[:, 0]), 2])
    for i in range(len(errorDeg[:, 0])):
        errorDeg[i, 0] = rError[i, 0]
        covDeg[i, 0] = rError[i, 0]
        errorDeg[i, 1] = np.arccos(np.dot(ref[i, 1:4], track[i, 1:4]))*180./np.pi
        covDeg[i, 1] = np.arccos(np.dot(ref[i, 1:4], track[i, 1:4]))
        covarVec = np.array(
            [track[i, 1] + np.sqrt(covar[i, 0,0]), track[i, 2] + np.sqrt(covar[i, 1,1]),
             track[i, 3] + np.sqrt(covar[i, 2,2])])
        covarVec = covarVec / np.linalg.norm(covarVec)
        covDeg[i, 1] = 3 * np.arccos(np.dot(covarVec, track[i, 1:4]))*180./np.pi

    t = ref[:, 0] * ns2min
    plt.figure(num=101011, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    plt.plot(t , errorDeg[:, 1], color = colorList[1] , label= "Off-point")
    plt.plot(t, covDeg[:,1], color=colorList[5], linestyle = '--', label=r'Covar (3-$\sigma$)')
    plt.legend(loc='upper right')
    plt.ylabel(r'Mean $\hat{\mathbf{h}}$ Error in $\mathcal{C}$ ($^\circ$)')
    plt.xlabel('Time (min)')
    plt.ylim([0,2])
    # plt.savefig('HeadingDeg.pdf')

    plt.figure(num=10101, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    plt.plot(t , rError[:, 1], color = colorList[2] , label= 'Error')
    plt.plot(t, 3*np.sqrt(covar[:, 0,0]), color=colorList[8], linestyle = '--', label=r'Covar (3-$\sigma$)')
    plt.plot(t, -3*np.sqrt(covar[:, 0,0]), color=colorList[8], linestyle = '--')
    plt.legend()
    plt.ylabel(r'$\hat{\mathbf{h}}_{' + str(1) +r'}$ Error in $\mathcal{C}$ (-)')
    plt.ylim([-0.04,0.04])
    plt.xlabel('Time (min)')
    # plt.savefig('HeadingCam1.pdf')

    plt.figure(num=10201, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    plt.plot(t , rError[:, 2], color = colorList[2] )
    plt.plot(t, 3*np.sqrt(covar[:, 1,1]), color=colorList[8], linestyle = '--')
    plt.plot(t, -3*np.sqrt(covar[:, 1,1]), color=colorList[8], linestyle = '--')
    plt.ylabel(r'$\hat{\mathbf{h}}_{' + str(2) +r'}$ Error in $\mathcal{C}$ (-)')
    plt.ylim([-0.04,0.04])
    plt.xlabel('Time (min)')
    # plt.savefig('HeadingCam2.pdf')

    plt.figure(num=10301, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    plt.plot(t , rError[:, 3], color = colorList[2])
    plt.plot(t, 3*np.sqrt(covar[:, 2,2]), color=colorList[8], linestyle = '--')
    plt.plot(t, -3*np.sqrt(covar[:, 2,2]), color=colorList[8], linestyle = '--')
    plt.ylabel(r'$\hat{\mathbf{h}}_{' + str(3) +r'}$ Error in $\mathcal{C}$ (-)')
    plt.ylim([-0.04,0.04])
    plt.xlabel('Time (min)')
    # plt.savefig('HeadingCam3.pdf')


def plot_faults(dataFaults, valid1, valid2):
    colorsInt = len(mpl.pyplot.get_cmap("inferno").colors)/(10.)
    colorList = []
    for i in range(10):
        colorList.append(mpl.pyplot.get_cmap("inferno").colors[int(i*colorsInt)])

    # for i in range(len(dataFaults[:,0])):
    #     if (dataFaults[i,0]*ns2min%1 != 0):
    #         valid1[i, 1] = np.nan
    #         valid2[i, 1] = np.nan

    plt.figure(10101, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    plt.xlabel('Time (min)')
    plt.ylabel('Detected Fault')
    plt.scatter(valid1[:,0]* ns2min, valid1[:,1], color = colorList[5], alpha = 0.1, label = "Limb")
    plt.scatter(valid2[:,0]* ns2min, valid2[:,1], color = colorList[8], alpha = 0.1, label = "Circ")
    plt.scatter(dataFaults[:,0]* ns2min, dataFaults[:,1], marker = '.', color = colorList[2], label = "Faults")
    plt.legend()
    # plt.savefig('FaultsDetected.pdf')
    return

def diff_methods(vec1, meth1, meth2, val1, val2):
    colorsInt = len(mpl.pyplot.get_cmap("inferno").colors)/(10.)
    colorList = []
    for i in range(10):
        colorList.append(mpl.pyplot.get_cmap("inferno").colors[int(i*colorsInt)])

    validIdx1 = []
    for i in range(len(val1[:,0])):
        if np.abs(val1[i,1] - 1) < 0.01:
            validIdx1.append(i)
    validIdx2 = []
    for i in range(len(val2[:,0])):
        if np.abs(val2[i,1] - 1) < 0.01:
            validIdx2.append(i)

    diff1 = np.full([len(validIdx1), 4], np.nan)
    diffNorms1 = np.full([len(validIdx1), 2], np.nan)
    diff2 = np.full([len(validIdx2), 4], np.nan)
    diffNorms2 = np.full([len(validIdx2), 2], np.nan)
    for i in range(len(validIdx1)):
        diff1[i,0] = vec1[validIdx1[i],0]
        diff1[i,1:] = vec1[validIdx1[i],1:] - meth1[validIdx1[i],1:]
        diffNorms1[i,0] = vec1[validIdx1[i],0]
        diffNorms1[i,1] = np.linalg.norm(vec1[validIdx1[i],1:]) - np.linalg.norm(meth1[validIdx1[i],1:])
    for i in range(len(validIdx2)):
        diff2[i,0] = vec1[validIdx2[i],0]
        diff2[i,1:] = vec1[validIdx2[i],1:] - meth2[validIdx2[i],1:]
        diffNorms2[i,0] = vec1[validIdx2[i],0]
        diffNorms2[i,1] = np.linalg.norm(vec1[validIdx2[i],1:]) - np.linalg.norm(meth2[validIdx2[i],1:])
    
    plt.figure(1, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    plt.xlabel('Time')
    plt.plot(diff1[:, 0] * ns2min, diff1[:, 1] * m2km, color = colorList[1], label=r"$\mathbf{r}_\mathrm{Limb}$")
    plt.plot(diff1[:, 0] * ns2min, diff1[:, 2] * m2km, color = colorList[5])
    plt.plot(diff1[:, 0] * ns2min, diff1[:, 3] * m2km, color = colorList[8])
    plt.plot(diff2[:, 0] * ns2min, diff2[:, 1] * m2km, color=colorList[1], label=r"$\mathbf{r}_\mathrm{Circ}$", linestyle ='--', linewidth=2)
    plt.plot(diff2[:, 0] * ns2min, diff2[:, 2] * m2km, color=colorList[5], linestyle ='--', linewidth=2)
    plt.plot(diff2[:, 0] * ns2min, diff2[:, 3] * m2km, color=colorList[8], linestyle ='--', linewidth=2)
    plt.legend()
    plt.ylabel(r"$\mathbf{r}_{\mathrm{true}} - \mathbf{r}_{\mathrm{opnav}}$ (km)")
    plt.xlabel("Time (min)")
    # plt.savefig('MeasErrorComponents.pdf')

    plt.figure(2, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    plt.xlabel('Time')
    plt.plot(diff1[:, 0] * ns2min, diffNorms1[:,1] * m2km, color = colorList[1])
    plt.plot(diff2[:, 0] * ns2min, diffNorms2[:,1] * m2km,  color = colorList[1], linestyle="--", linewidth=2)
    plt.ylabel(r"$|\mathbf{r}_{\mathrm{true}}|$ - $|\mathbf{r}_{\mathrm{opnav}}|$ (km)")
    plt.xlabel("Time (min)")
    # plt.savefig('MeasErrorNorm.pdf')

def diff_vectors(vec1, vec2, valid, string):
    assert len(vec1[0,:]) == len(vec2[0,:]), print("Vectors need to be the same size")

    colorsInt = len(mpl.pyplot.get_cmap("inferno").colors)/(10.)
    colorList = []
    for i in range(10):
        colorList.append(mpl.pyplot.get_cmap("inferno").colors[int(i*colorsInt)])

    validIdx = []
    for i in range(len(valid[:,0])):
        if np.abs(valid[i,1] - 1) < 0.01:
            validIdx.append(i)
    diff = np.full([len(validIdx), 4], np.nan)
    diffNorms = np.full([len(validIdx), 2], np.nan)
    m2km2 = m2km*m2km
    for i in range(len(validIdx)):
        diff[i,0] = vec1[validIdx[i],0]
        diff[i,1:] = vec1[validIdx[i],1:] - vec2[validIdx[i],1:]
        diffNorms[i,0] = vec1[validIdx[i],0]
        diffNorms[i,1] = np.linalg.norm(vec1[validIdx[i],1:]) - np.linalg.norm(vec2[validIdx[i],1:])
    # # plt.figure(1, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    # plt.figure(1, figsize=(3.5, 2.), facecolor='w', edgecolor='k')
    # # plt.xlabel('Time')
    # plt.plot(diff[:, 0] * ns2min, diff[:, 1] * m2km2, color = colorList[1], label=r"$x_\mathrm{"+string+"}$")
    # plt.plot(diff[:, 0] * ns2min, np.mean(diff[:, 1]) * m2km2 * np.ones(len(diff[:, 0])), color = colorList[1], linestyle = '--')
    # plt.plot(diff[:, 0] * ns2min, diff[:, 2] * m2km2, color = colorList[5], label=r"$y_\mathrm{"+string+"}$")
    # plt.plot(diff[:, 0] * ns2min, np.mean(diff[:, 2]) * m2km2 * np.ones(len(diff[:, 0])), color = colorList[5], linestyle = '--')
    # plt.plot(diff[:, 0] * ns2min, diff[:, 3] * m2km2, color = colorList[8], label=r"$z_\mathrm{"+string+"}$")
    # plt.plot(diff[:, 0] * ns2min, np.mean(diff[:, 3]) * m2km2 * np.ones(len(diff[:, 0])), color = colorList[8], linestyle = '--')
    # plt.legend()
    # plt.ylabel(r"$\mathbf{r}_{\mathrm{true}} - \mathbf{r}_{\mathrm{opnav}}$ (km)")
    # plt.xlabel("Time (min)")
    # plt.grid(True)
    # ##plt.savefig('MeasErrorComponents.pdf')

    # # plt.figure(2, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    # plt.figure(2, figsize=(3.5, 2.), facecolor='w', edgecolor='k')
    # plt.plot(diff[:, 0] * ns2min, diffNorms[:,1] * m2km2, color = colorList[1])
    # plt.plot(diff[:, 0] * ns2min, np.mean(diffNorms[:,1]) * m2km2 * np.ones(len(diff[:, 0])),  color = colorList[1], linestyle="--")
    # plt.ylabel(r"$|\mathbf{r}_{\mathrm{true}}|$ - $|\mathbf{r}_{\mathrm{opnav}}|$ (km)")
    # plt.xlabel("Time (min)")
    # plt.grid(True)
    # #plt.savefig('MeasErrorNorm.pdf')

    fig, axes = plt.subplots(2, 1, figsize=(6, 5.5))

    # -------------------------
    # Position error components
    # -------------------------
    ax = axes[0]

    ax.plot(diff[:, 0] * ns2min,
            diff[:, 1] * m2km2,
            color=colorList[1],
            label=r"$x_{\mathrm{true}}-x_{\mathrm{opnav}}$")

    ax.plot(diff[:, 0] * ns2min,
            diff[:, 2] * m2km2,
            color=colorList[5],
            label=r"$y_{\mathrm{true}}-y_{\mathrm{opnav}}$")

    ax.plot(diff[:, 0] * ns2min,
            diff[:, 3] * m2km2,
            color=colorList[8],
            label=r"$z_{\mathrm{true}}-z_{\mathrm{opnav}}$")

    ax.set_ylabel("Position Error (km)")
    ax.set_title("Position Error Components")
    ax.legend(loc='best')
    ax.grid(True)

    # -------------------------
    # Radial distance difference
    # -------------------------
    ax = axes[1]

    ax.plot(diffNorms[:, 0] * ns2min,
            diffNorms[:, 1] * m2km2,
            color=colorList[1])

    ax.set_ylabel(
        r"$|\mathbf{r}_{\mathrm{true}}|-|\mathbf{r}_{\mathrm{opnav}}|$ (km)"
    )
    ax.set_xlabel("Time (min)")
    ax.set_title("Radial Distance Difference")
    ax.grid(True)

    fig.suptitle("Navigation Position Errors")
    plt.tight_layout(rect=[0, 0, 1, 0.94])


    return

def nav_percentages(truth, states, covar, valid, string):
    numStates = len(states[0:,1:])
    colorsInt = len(mpl.pyplot.get_cmap("inferno").colors)/(10.)
    colorList = []
    for i in range(10):
        colorList.append(mpl.pyplot.get_cmap("inferno").colors[int(i*colorsInt)])

    validIdx = []
    for i in range(len(valid[:,0])):
        if np.abs(valid[i,1] - 1) < 0.01:
            validIdx.append(i)
    diffPos = np.full([len(validIdx), 2], np.nan)
    diffVel = np.full([len(validIdx), 2], np.nan)
    covarPos = np.full([len(validIdx), 2], np.nan)
    covarVel = np.full([len(validIdx), 2], np.nan)

    m2km2 = m2km
    for i in range(len(validIdx)):
        diffPos[i,0] = states[validIdx[i],0]
        diffPos[i,1] = np.linalg.norm(states[validIdx[i],1:4] - truth[validIdx[i],1:4])/np.linalg.norm(truth[validIdx[i],1:4])*100
        diffVel[i,0] = states[validIdx[i],0]
        diffVel[i,1] = np.linalg.norm(states[validIdx[i],4:7] - truth[validIdx[i],4:7])/np.linalg.norm(truth[validIdx[i],4:7])*100
        covarPos[i,0] = states[validIdx[i],0]
        posVec = np.sqrt(np.array([covar[validIdx[i],1], covar[validIdx[i],1 + 6+1], covar[validIdx[i],1 + 2*(6+1)]]))
        covarPos[i,1] =  3*np.linalg.norm(posVec)/np.linalg.norm(truth[validIdx[i],1:4])*100
        covarVel[i,0] = states[validIdx[i],0]
        velVec = np.sqrt(np.array([covar[validIdx[i],1 + 3*(6+1)], covar[validIdx[i],1 + 4*(6+1)], covar[validIdx[i],1 + 5*(6+1)]]))
        covarVel[i,1] = 3*np.linalg.norm(velVec)/np.linalg.norm(truth[validIdx[i],4:7])*100
    # plt.figure(101, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    # # plt.figure(101, figsize=(3.5, 2), facecolor='w', edgecolor='k')
    # plt.plot(diffPos[:, 0] * ns2min, diffPos[:, 1] , color = colorList[1], label = "Error")
    # plt.plot(covarPos[:, 0] * ns2min, covarPos[:,1], color = colorList[8], linestyle = '--', label=r'Covar ($3\sigma$)')
    # plt.legend(loc='upper right')
    # plt.ylabel(r"$\mathbf{r}_\mathrm{"+string+r"}$ errors ($\%$)")
    # plt.xlabel("Time (min)")
    # # plt.ylim([0,3.5])
    # #plt.savefig('PercentErrorPos.pdf')

    # plt.figure(102, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    # # plt.figure(102, figsize=(3.5, 2.), facecolor='w', edgecolor='k')
    # plt.plot(diffVel[:, 0] * ns2min, diffVel[:, 1], color = colorList[1])
    # plt.plot(covarVel[:, 0] * ns2min, covarVel[:,1], color = colorList[8], linestyle = '--')
    # plt.ylabel(r"$\dot{\mathbf{r}}_\mathrm{"+string+ r"}$ errors ($\%$)")
    # plt.xlabel("Time (min)")
    # # plt.ylim([0,15])
    # #plt.savefig('PercentErrorVel.pdf')

    fig, axes = plt.subplots(2, 1, figsize=(6, 5.5),
                             sharex=True)

    # -------------------------
    # Position percentage error
    # -------------------------
    ax = axes[0]

    ax.plot(
        diffPos[:, 0] * ns2min,
        diffPos[:, 1],
        color=colorList[1],
        label="Error"
    )

    ax.plot(
        covarPos[:, 0] * ns2min,
        covarPos[:, 1],
        color=colorList[8],
        linestyle='--',
        label=r'Covar ($3\sigma$)'
    )

    ax.set_ylabel(r"$\mathbf{r}_{\mathrm{"+string+"}}$ error (%)")
    ax.set_title("Position")
    ax.legend(loc='best')
    ax.grid(True)

    # -------------------------
    # Velocity percentage error
    # -------------------------
    ax = axes[1]

    ax.plot(
        diffVel[:, 0] * ns2min,
        diffVel[:, 1],
        color=colorList[1],
        label="Error"
    )

    ax.plot(
        covarVel[:, 0] * ns2min,
        covarVel[:, 1],
        color=colorList[8],
        linestyle='--',
        label=r'Covar ($3\sigma$)'
    )

    ax.set_ylabel(r"$\dot{\mathbf{r}}_{\mathrm{"+string+"}}$ error (%)")
    ax.set_xlabel("Time (min)")
    ax.set_title("Velocity")
    ax.legend(loc='best')
    ax.grid(True)

    fig.suptitle("Navigation Error Percentages")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    RMSPos = np.sqrt(sum(diffPos[:, 1] ** 2)/len(diffPos[:, 1]))
    RMSPosCov = np.sqrt(sum((covarPos[:, 1]) ** 2)/len(covarPos[:, 1]))
    RMSVel = np.sqrt(sum(diffVel[:, 1] ** 2)/len(diffVel[:, 1]))
    RMSVelCov = np.sqrt(sum((covarVel[:, 1]) ** 2)/len(covarVel[:, 1]))

    print('-------- Pos RMS '+ string + '--------')
    print(RMSPos)
    print(RMSPosCov)
    print('------------------------')
    print('-------- Vel RMS '+ string + '--------')
    print(RMSVel)
    print(RMSVelCov)
    print('------------------------')
    # RMS Errors Computed for heading and rate
    # simHelpers.writeTeXSnippet('RMSPos_'+ string, str(round(RMSPos,3)), os.path.dirname(__file__))
    # simHelpers.writeTeXSnippet('RMSPosCov_'+ string, str(round(RMSPosCov,3)),os.path.dirname(__file__))
    # simHelpers.writeTeXSnippet('RMSVel_'+ string, str(round(RMSVel,3)), os.path.dirname(__file__))
    # simHelpers.writeTeXSnippet('RMSVelCov_'+ string, str(round(RMSVelCov,3)),os.path.dirname(__file__))
    return

def plot_orbit(r_BN):
    plt.figure()
    plt.xlabel('$R_x$, km')
    plt.ylabel('$R_y$, km')
    plt.plot(r_BN[:, 1] * m2km, r_BN[:, 2] * m2km, color_x)
    plt.scatter(0, 0)
    plt.title('Spacecraft Orbit')
    return

def plot_rw_motor_torque(timeData, dataUsReq, dataRW, numRW):
    colorsInt = len(mpl.pyplot.get_cmap("inferno").colors)/(10.)
    colorList = []
    for i in range(10):
        colorList.append(mpl.pyplot.get_cmap("inferno").colors[int(i*colorsInt)])
    plt.figure(22, figsize=(3, 1.8), facecolor='w', edgecolor='k')
    for idx in range(1, numRW):
        plt.plot(timeData, dataUsReq[:, idx],
                 '--',
                 color=colorList[idx*2],
                 label=r'$\hat u_{s,' + str(idx) + '}$')
        plt.plot(timeData, dataRW[idx - 1][:, 1],
                 color=colorList[idx*2],
                 label='$u_{s,' + str(idx) + '}$')
    plt.legend(loc='lower right')
    #plt.savefig('RWMotorTorque.pdf')
    plt.xlabel('Time (min)')
    plt.ylabel('RW Motor Torque (Nm)')


def plot_TwoOrbits(r_BN, r_BN2):
    colorsInt = len(mpl.pyplot.get_cmap("inferno").colors)/(10.)
    colorList = []
    for i in range(10):
        colorList.append(mpl.pyplot.get_cmap("inferno").colors[int(i*colorsInt)])

    # fig = plt.figure(5, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    fig = plt.figure(5, figsize=(3.5, 2.), facecolor='w', edgecolor='k')
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel(r'$R_x$, km')
    ax.set_ylabel(r'$R_y$, km')
    ax.set_zlabel(r'$R_z$, km')
    ax.set_title('True and OpNav-Measured Spacecraft Position')
    ax.legend(loc='best')
    ax.plot(r_BN[:, 1] * m2km, r_BN[:, 2] * m2km, r_BN[:, 3] * m2km, color=colorList[1], label="True")
    for i in range(len(r_BN2[:,0])):
        if np.abs(r_BN2[i, 1])>0 or np.abs(r_BN2[i, 2])>0:
            ax.scatter(r_BN2[i, 1] * m2km, r_BN2[i, 2] * m2km, r_BN2[i, 3] * m2km, color=colorList[8], label="Measured")
    ax.scatter(0, 0, color='r')
    # ax.set_title('Orbit and Measurements')
    return

def plot_attitude_error(timeLineSet, dataSigmaBR):
    colorsInt = len(mpl.pyplot.get_cmap("inferno").colors)/(10.)
    colorList = []
    for i in range(10):
        colorList.append(mpl.pyplot.get_cmap("inferno").colors[int(i*colorsInt)])

    plt.figure(5555, figsize=(3, 1.8), facecolor='w', edgecolor='k')
    plt.rcParams["font.size"] = "8"
    fig = plt.gcf()
    ax = fig.gca()
    vectorData = simHelpers.pullVectorSetFromData(dataSigmaBR)
    sNorm = np.array([np.linalg.norm(v) for v in vectorData])
    plt.plot(timeLineSet, sNorm,
             color=colorList[0],
             )
    plt.xlabel('Time (min)')
    plt.xlim([40, 100])
    plt.ylabel(r'Attitude Error Norm $|\sigma_{C/R}|$')
    #plt.savefig('AttErrorNorm.pdf')
    ax.set_yscale('log')


def plot_rate_error(timeLineSet, dataOmegaBR):
    colorsInt = len(mpl.pyplot.get_cmap("inferno").colors)/(10.)
    colorList = []
    for i in range(10):
        colorList.append(mpl.pyplot.get_cmap("inferno").colors[int(i*colorsInt)])
    plt.figure(666666, figsize=(3, 1.8), facecolor='w', edgecolor='k')
    plt.rcParams["font.size"] = "8"
    styleList = ['-', '--', ':']
    for idx in range(1, 4):
        plt.plot(timeLineSet, dataOmegaBR[:, idx],
                 color=colorList[idx*2],
                 label=r'$\omega_{BR,' + str(idx) + '}$',
                 linestyle= styleList[idx-1] )
    plt.legend(loc='lower right')
    plt.xlim([40, 100])
    plt.xlabel('Time (min)')
    #plt.savefig('RateErrorControl.pdf')
    plt.ylabel('Rate Tracking Error (rad/s) ')
    return


def plot_rw_cmd_torque(timeData, dataUsReq, numRW):
    plt.figure()
    for idx in range(1, 4):
        plt.plot(timeData, dataUsReq[:, idx],
                 '--',
                 color=simHelpers.getLineColor(idx, numRW),
                 label=r'$\hat u_{s,' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time (min)')
    #plt.savefig('RWMotorTorque.pdf')
    plt.ylabel('RW Motor Torque (Nm)')

def plot_rw_speeds(timeData, dataOmegaRW, numRW):
    plt.figure()
    for idx in range(1, numRW + 1):
        plt.plot(timeData, dataOmegaRW[:, idx] / mc.RPM,
                 color=simHelpers.getLineColor(idx, numRW),
                 label=r'$\Omega_{' + str(idx) + '}$')
    plt.legend(loc='upper right')
    plt.xlabel('Time [min]')
    plt.ylabel('RW Speed (RPM) ')

def plotStateCovarPlot(x, Pflat):

    colorsInt = len(mpl.pyplot.get_cmap("inferno").colors)/(10.)
    colorList = []
    for i in range(10):
        colorList.append(mpl.pyplot.get_cmap("inferno").colors[int(i*colorsInt)])

    numStates = len(x[0,:])-1

    P = np.zeros([len(Pflat[:,0]),numStates,numStates])
    t= np.zeros(len(Pflat[:,0]))
    for i in range(len(Pflat[:,0])):
        t[i] = x[i, 0]*ns2min
        if not np.isnan(Pflat[i,1]):
            Plast = Pflat[i,1:].reshape([numStates,numStates])
            P[i, :, :] = Plast
            x0 = x[i,7:10]
        if numStates == 6 or numStates == 3:
            P[i,:,:] = Pflat[i,1:].reshape([numStates,numStates])


    # if numStates == 9 :
    #     plt.figure(10, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    #     plt.plot(t , x[:, 1]*m2km, label=r'$r_1$ error',color = colorList[1])
    #     plt.plot(t , 3 * np.sqrt(P[:, 0, 0])*m2km, '--',  label=r'$\pm3\sigma$', color = colorList[8])
    #     plt.plot(t ,- 3 * np.sqrt(P[:, 0, 0])*m2km, '--', color = colorList[8])
    #     plt.legend(loc='best')
    #     plt.xlabel('Time (min)')
    #     plt.ylabel('Position Error (km)')
    #     plt.title('First position component')
    #     plt.grid()

    #     plt.figure(11, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    #     plt.plot(t , x[:, 4]*m2km, label=r'$v_1$ error',color = colorList[1])
    #     plt.plot(t , 3 * np.sqrt(P[:, 3, 3])*m2km, '--', label=r'$\pm3\sigma$',color = colorList[8])
    #     plt.plot(t ,- 3 * np.sqrt(P[:, 3, 3])*m2km, '--', color = colorList[8])
    #     plt.xlabel('Time (min)')
    #     plt.ylabel('Rate Error (km/s)')
    #     plt.title('First rate component')
    #     plt.grid()

    #     plt.figure(12, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    #     plt.plot(t , x[:, 7], label='Bias error',color = colorList[1])
    #     plt.plot(t , x0[0] + 3 * np.sqrt(P[:, 6, 6]), '--', label=r'$\pm3\sigma$',color = colorList[8])
    #     plt.plot(t , x0[0] - 3 * np.sqrt(P[:, 6, 6]), '--', color = colorList[8])
    #     plt.title('First bias component')
    #     plt.ylabel('Bias Error (km/s)')
    #     plt.xlabel('Time (min)')
    #     plt.grid()

    #     plt.figure(13, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    #     plt.plot(t , x[:, 2]*m2km,  label=r'$r_2$ error',color = colorList[1])
    #     plt.plot(t , 3 * np.sqrt(P[:, 1, 1])*m2km, '--', label=r'$\pm3\sigma$',color = colorList[8])
    #     plt.plot(t ,- 3 * np.sqrt(P[:, 1, 1])*m2km, '--', color = colorList[8])
    #     plt.xlabel('Time (min)')
    #     plt.ylabel('Position Error (km)')
    #     plt.title('Second position component')
    #     plt.grid()

    #     plt.figure(14, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    #     plt.plot(t , x[:, 5]*m2km, label=r'$v_2$ error',color = colorList[1])
    #     plt.plot(t , 3 * np.sqrt(P[:, 4, 4])*m2km, '--', label=r'$\pm3\sigma$',color = colorList[8])
    #     plt.plot(t ,- 3 * np.sqrt(P[:, 4, 4])*m2km, '--', color = colorList[8])
    #     plt.xlabel('Time (min)')
    #     plt.ylabel('Rate Error (km/s)')
    #     plt.title('Second rate component')
    #     plt.grid()

    #     plt.figure(15, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    #     plt.plot(t , x[:, 8], label='Bias error',color = colorList[1])
    #     plt.plot(t , x0[1] + 3 * np.sqrt(P[:, 7, 7]), '--', label=r'$\pm3\sigma$',color = colorList[8])
    #     plt.plot(t , x0[1] - 3 * np.sqrt(P[:, 7, 7]), '--', color = colorList[8])
    #     plt.title('Second bias component')
    #     plt.ylabel('Bias Error (km/s)')
    #     plt.xlabel('Time (min)')
    #     plt.grid()

    #     plt.figure(16, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    #     plt.plot(t , x[:, 3]*m2km,  label=r'$r_3$ error',color = colorList[1])
    #     plt.plot(t , 3 * np.sqrt(P[:, 2, 2])*m2km, '--',label=r'$\pm3\sigma$', color = colorList[8])
    #     plt.plot(t ,-3 * np.sqrt(P[:, 2, 2])*m2km, '--', color = colorList[8])
    #     plt.xlabel('Time (min)')
    #     plt.ylabel('Position Error (km)')
    #     plt.title('Third position component')
    #     plt.grid()

    #     plt.figure(17, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    #     plt.plot(t , x[:, 6]*m2km,label=r'$v_3$ error', color = colorList[1])
    #     plt.plot(t , 3 * np.sqrt(P[:, 5, 5])*m2km, '--',label=r'$\pm3\sigma$', color = colorList[8])
    #     plt.plot(t , -3 * np.sqrt(P[:, 5, 5])*m2km, '--', color = colorList[8])
    #     plt.xlabel('Time (min)')
    #     plt.ylabel('Rate Error (km/s)')
    #     plt.title('Third rate component')
    #     plt.grid()

    #     plt.figure(18, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    #     plt.plot(t , x[:, 9],label='Bias error', color = colorList[1])
    #     plt.plot(t , x0[2] + 3 * np.sqrt(P[:, 8, 8]), '--',label=r'$\pm3\sigma$', color = colorList[8])
    #     plt.plot(t , x0[2] - 3 * np.sqrt(P[:, 8, 8]), '--', color = colorList[8])
    #     plt.title('Third bias component')
    #     plt.ylabel('Bias Error (km/s)')
    #     plt.xlabel('Time (min)')
    #     plt.grid()
    
    if numStates == 9:
        # Create a 3x3 figure:
        # Row 1: Position errors
        # Row 2: Velocity errors
        # Row 3: Bias errors
        fig, axes = plt.subplots(3, 3, figsize=(10, 7.5), sharex=True)

        # State/error indices in x
        position_idx = [1, 2, 3]
        velocity_idx = [4, 5, 6]
        bias_idx = [7, 8, 9]

        # Corresponding covariance diagonal indices
        position_P_idx = [0, 1, 2]
        velocity_P_idx = [3, 4, 5]
        bias_P_idx = [6, 7, 8]

        # Titles
        position_titles = [r'Position $x$', r'Position $y$', r'Position $z$']
        velocity_titles = [r'Velocity $x$', r'Velocity $y$', r'Velocity $z$']
        bias_titles = [r'Bias $x$', r'Bias $y$', r'Bias $z$']

        # Plot position, velocity and bias states
        for i in range(3):

            # -------------------------
            # Position
            # -------------------------
            ax = axes[0, i]

            ax.plot(t,
                    x[:, position_idx[i]] * m2km,
                    color=colorList[1],
                    label='State error')
            ax.plot(t,
                    3 * np.sqrt(P[:, position_P_idx[i],
                                   position_P_idx[i]]) * m2km,
                    '--',
                    color=colorList[8],
                    label=r'$\pm3\sigma$')
            ax.plot(t,
                    -3 * np.sqrt(P[:, position_P_idx[i],
                                    position_P_idx[i]]) * m2km,
                    '--',
                    color=colorList[8])

            ax.set_title(position_titles[i])
            ax.set_ylabel('Error (km)')
            ax.grid(True)

            # -------------------------
            # Velocity
            # -------------------------
            ax = axes[1, i]

            ax.plot(t,
                    x[:, velocity_idx[i]] * m2km,
                    color=colorList[1],
                    label='State error')
            ax.plot(t,
                    3 * np.sqrt(P[:, velocity_P_idx[i],
                                   velocity_P_idx[i]]) * m2km,
                    '--',
                    color=colorList[8],
                    label=r'$\pm3\sigma$')
            ax.plot(t,
                    -3 * np.sqrt(P[:, velocity_P_idx[i],
                                    velocity_P_idx[i]]) * m2km,
                    '--',
                    color=colorList[8])

            ax.set_title(velocity_titles[i])
            ax.set_ylabel('Error (km/s)')
            ax.grid(True)

            # -------------------------
            # Bias
            # -------------------------
            ax = axes[2, i]

            ax.plot(t,
                    x[:, bias_idx[i]],
                    color=colorList[1],
                    label='State error')
            ax.plot(t,
                    x0[i] + 3 * np.sqrt(P[:, bias_P_idx[i],
                                           bias_P_idx[i]]),
                    '--',
                    color=colorList[8],
                    label=r'$\pm3\sigma$')
            ax.plot(t,
                    x0[i] - 3 * np.sqrt(P[:, bias_P_idx[i],
                                           bias_P_idx[i]]),
                    '--',
                    color=colorList[8])

            ax.set_title(bias_titles[i])
            ax.set_ylabel('Error (km/s)')
            ax.grid(True)

        # X-axis labels only on bottom row
        for ax in axes[2, :]:
            ax.set_xlabel('Time (min)')

        # One common legend for the entire figure
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels,
                   loc='upper center',
                   ncol=2,
                   bbox_to_anchor=(0.5, 1.02))

        fig.suptitle('State Estimation Errors and 3-$\sigma$ Covariance',
                     y=1.06)

        # Adjust spacing
        plt.tight_layout()
        
    # if numStates == 6 or numStates ==3:
    #     plt.figure(20, figsize=(2.4, 1.4), facecolor='w', edgecolor='k')
    #     plt.plot(t , x[:, 1]*m2km, label='State Error', color = colorList[1])
    #     plt.plot(t , 3 * np.sqrt(P[:, 0, 0])*m2km, '--',  label=r'$\pm3\sigma$',color = colorList[8])
    #     plt.plot(t ,- 3 * np.sqrt(P[:, 0, 0])*m2km, '--', color = colorList[8])
    #     plt.legend(loc='best')
    #     plt.ylabel(r'$r_1$ Error (km)')
    #     #plt.savefig('Filterpos1.pdf')

    # if numStates == 6:
    #     plt.figure(21, figsize=(2.4, 1.4), facecolor='w', edgecolor='k')
    #     plt.plot(t , x[:, 4]*m2km, color = colorList[1])
    #     plt.plot(t , 3 * np.sqrt(P[:, 3, 3])*m2km, '--', label=r'$\pm3\sigma$',color = colorList[8])
    #     plt.plot(t ,- 3 * np.sqrt(P[:, 3, 3])*m2km, '--', color = colorList[8])
    #     plt.ylabel(r'$v_1$ Error (km/s)')
    #     #plt.savefig('Filtervel1.pdf')

    # if numStates == 6 or numStates ==3:
    #     plt.figure(22, figsize=(2.4, 1.4), facecolor='w', edgecolor='k')
    #     plt.plot(t , x[:, 2]*m2km, color = colorList[1])
    #     plt.plot(t , 3 * np.sqrt(P[:, 1, 1])*m2km, '--', label=r'$\pm3\sigma$',color = colorList[8])
    #     plt.plot(t ,- 3 * np.sqrt(P[:, 1, 1])*m2km, '--', color = colorList[8])
    #     plt.ylabel(r'$r_2$ Error (km)')
    #     #plt.savefig('Filterpos2.pdf')

    # if numStates == 6:
    #     plt.figure(23, figsize=(2.4, 1.4), facecolor='w', edgecolor='k')
    #     plt.plot(t , x[:, 5]*m2km, color = colorList[1])
    #     plt.plot(t , 3 * np.sqrt(P[:, 4, 4])*m2km, '--',label=r'$\pm3\sigma$', color = colorList[8])
    #     plt.plot(t ,- 3 * np.sqrt(P[:, 4, 4])*m2km, '--', color = colorList[8])
    #     plt.ylabel(r'$v_2$ Error (km/s)')
    #     #plt.savefig('Filtervel2.pdf')

    # if numStates == 6 or numStates ==3:
    #     plt.figure(24, figsize=(2.4, 1.4), facecolor='w', edgecolor='k')
    #     plt.plot(t , x[:, 3]*m2km, color = colorList[1])
    #     plt.plot(t , 3 * np.sqrt(P[:, 2, 2])*m2km, '--',label=r'$\pm3\sigma$', color = colorList[8])
    #     plt.plot(t ,-3 * np.sqrt(P[:, 2, 2])*m2km, '--', color = colorList[8])
    #     plt.xlabel('Time (min)')
    #     plt.ylabel(r'$r_3$ Error (km)')
    #     #plt.savefig('Filterpos3.pdf')

    # if numStates == 6:
    #     plt.figure(25, figsize=(2.4, 1.4), facecolor='w', edgecolor='k')
    #     plt.plot(t , x[:, 6]*m2km, color = colorList[1])
    #     plt.plot(t , 3 * np.sqrt(P[:, 5, 5])*m2km, '--',label=r'$\pm3\sigma$', color = colorList[8])
    #     plt.plot(t , -3 * np.sqrt(P[:, 5, 5])*m2km, '--', color = colorList[8])
    #     plt.xlabel('Time (min)')
    #     plt.ylabel(r'$v_3$ Error (km/s)')
    #     #plt.savefig('Filtervel3.pdf')

    if numStates == 6:
        # Create a 2x3 figure:
        # Row 1: Position errors
        # Row 2: Velocity errors
        fig, axes = plt.subplots(2, 3, figsize=(10, 5.5),
                                 sharex=True)

        # State/error indices in x
        position_idx = [1, 2, 3]
        velocity_idx = [4, 5, 6]

        # Corresponding covariance diagonal indices
        position_P_idx = [0, 1, 2]
        velocity_P_idx = [3, 4, 5]

        # Titles
        position_titles = [
            r'Position $x$',
            r'Position $y$',
            r'Position $z$'
        ]

        velocity_titles = [
            r'Velocity $x$',
            r'Velocity $y$',
            r'Velocity $z$'
        ]

        for i in range(3):

            # -------------------------
            # Position
            # -------------------------
            ax = axes[0, i]

            ax.plot(
                t,
                x[:, position_idx[i]] * m2km,
                color=colorList[1],
                label='State error'
            )

            sigma = 3 * np.sqrt(
                P[:, position_P_idx[i], position_P_idx[i]]
            ) * m2km

            ax.plot(
                t, sigma,
                '--',
                color=colorList[8],
                label=r'$\pm3\sigma$'
            )

            ax.plot(
                t, -sigma,
                '--',
                color=colorList[8]
            )

            ax.set_title(position_titles[i])
            ax.set_ylabel('Error (km)')
            ax.grid(True)

            # -------------------------
            # Velocity
            # -------------------------
            ax = axes[1, i]

            ax.plot(
                t,
                x[:, velocity_idx[i]] * m2km,
                color=colorList[1],
                label='State error'
            )

            sigma = 3 * np.sqrt(
                P[:, velocity_P_idx[i], velocity_P_idx[i]]
            ) * m2km

            ax.plot(
                t, sigma,
                '--',
                color=colorList[8],
                label=r'$\pm3\sigma$'
            )

            ax.plot(
                t, -sigma,
                '--',
                color=colorList[8]
            )

            ax.set_title(velocity_titles[i])
            ax.set_ylabel('Error (km/s)')
            ax.set_xlabel('Time (min)')
            ax.grid(True)

        # One common legend for the entire figure
        handles, labels = axes[0, 0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc='lower center',
            ncol=2,
            bbox_to_anchor=(0.5, 1.02)
        )

        fig.suptitle(
            'State Estimation Errors and 3-$\sigma$ Covariance',
            y=0.98
        )

        plt.tight_layout(rect=[0, 0.07, 1, 0.94])

    if numStates == 3:

        fig, axes = plt.subplots(3, 1, figsize=(6, 7),
                                 sharex=True)

        position_idx = [1, 2, 3]
        covariance_idx = [0, 1, 2]

        position_titles = [
            r'Position $x$',
            r'Position $y$',
            r'Position $z$'
        ]

        for i in range(3):

            ax = axes[i]

            sigma = 3 * np.sqrt(
                P[:, covariance_idx[i], covariance_idx[i]]
            ) * m2km

            ax.plot(
                t,
                x[:, position_idx[i]] * m2km,
                color=colorList[1],
                label='State error'
            )

            ax.plot(
                t,
                sigma,
                '--',
                color=colorList[8],
                label=r'$\pm3\sigma$'
            )

            ax.plot(
                t,
                -sigma,
                '--',
                color=colorList[8]
            )

            ax.set_title(position_titles[i])
            ax.set_ylabel('Error (km)')
            ax.grid(True)

        axes[-1].set_xlabel('Time (min)')

        handles, labels = axes[0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc='upper center',
            ncol=2,
            bbox_to_anchor=(0.5, 1.01)
        )

        fig.suptitle(
            'Position State Errors and 3-$\sigma$ Covariance',
            y=1.05
        )

        plt.tight_layout()

def centerXY(centerPoints, size):

    t= centerPoints[:, 0]*ns2min
    subtime = []
    subX = []
    subY = []

    center = np.full([len(t), 3], np.nan)
    for i in range(len(centerPoints[:,0])):
        if centerPoints[i, 1:3].any() > 1E-10:
            subtime.append(centerPoints[i, 0]*ns2min)
            center[i,1:] = centerPoints[i,1:3] - size/2 + 0.5
            subX.append(center[i,1])
            subY.append(center[i, 2])

    colorsInt = len(mpl.cm.get_cmap("inferno").colors)/(10.)
    colorList = []
    for i in range(10):
        colorList.append(mpl.cm.get_cmap("inferno").colors[int(i*colorsInt)])

    # xCurve = fit_sin(subtime, subX)
    # yCurve = fit_sin(subtime, subY)
    #
    # print "Periods in Errors are , " + str(xCurve["period"]) + " and " + str(yCurve["period"]) + " s"

    # plt.figure(100, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    plt.figure(100, figsize=(3.5, 2.), facecolor='w', edgecolor='k')
    plt.plot(subtime, subX, ".", label = "X-center", color = colorList[8])
    # plt.plot(t, xCurve["fitfunc"](t), "r", label="X-curve")
    plt.plot(subtime, subY, ".", label = "Y-center", color = colorList[1])
    # plt.plot(t, yCurve["fitfunc"](t), "b", label="Y-curve")
    plt.legend(loc='best')
    plt.ylabel("Pixels")
    plt.title('First unit Vec component in C frame (-)')
    #plt.savefig('CentersXY.pdf')


def pixelAndPos(x, r, centers, size):

    t= x[:, 0]*ns2min
    r_norm = np.zeros([len(r[:,0]), 5])
    r_norm[:,0] = r[:,0]
    centerPoints = np.full([len(t), 3], np.nan)
    maxDiff = np.zeros(3)
    for i in range(3):
        values = []
        for j in range(len(x[:, 0])):
            if not np.isnan(x[j,i+1]):
                values.append(x[j,i+1])
        try:
            maxDiff[i] = np.max(np.abs(values))
        except ValueError:
            maxDiff[i] =1
    for i in range(len(x[:,0])):
        r_norm[i,1:4] = r[i,1:]/np.linalg.norm(r[i,1:]) * maxDiff
        r_norm[i,4] = np.linalg.norm(r[i,1:])
        if centers[i, 1:3].any() > 1E-10:
            centerPoints[i,1:] = centers[i,1:3] - size/2
    for i in range(2):
        values = []
        for j in range(len(centerPoints[:, 0])):
            if not np.isnan(centerPoints[j, i + 1]):
                values.append(centerPoints[j, i + 1])
        try:
            maxCenter = np.max(np.abs(values))
        except ValueError:
            maxCenter = 1
        centerPoints[:, i+1]/=maxCenter/maxDiff[i]


    colorsInt = len(mpl.cm.get_cmap("inferno").colors)/(10.)
    colorList = []
    for i in range(10):
        colorList.append(mpl.cm.get_cmap("inferno").colors[int(i*colorsInt)])

    # plt.figure(200, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    plt.figure(200, figsize=(3.5, 2.), facecolor='w', edgecolor='k')
    plt.plot(t, x[:, 1], ".", label='Truth - Meas', color = colorList[1])
    plt.plot(t, r_norm[:, 1], label = "True Pos", color = colorList[5])
    plt.plot(t, centerPoints[:,1], ".", label = "X-center", color = colorList[8])
    # plt.plot(t, x1["fitfunc"](t), "r--", label = "Fit")
    plt.legend(loc='best')
    plt.title('First unit Vec component in C frame (-)')

    # plt.figure(201, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    plt.figure(201, figsize=(3.5, 2.), facecolor='w', edgecolor='k')
    plt.plot(t, x[:, 2], ".", color = colorList[1])
    plt.plot(t, r_norm[:, 2], color = colorList[5])
    plt.plot(t, centerPoints[:,2], ".", label = "X-center", color = colorList[8])
    # plt.plot(t, x2["fitfunc"](t), "r--")
    plt.title('Second unit Vec component in C frame (-)')

    # plt.figure(202, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    plt.figure(202, figsize=(3.5, 2.), facecolor='w', edgecolor='k')
    plt.plot(t, x[:, 3], ".", color = colorList[1])
    plt.plot(t, r_norm[:, 3], color = colorList[5])
    # plt.plot(t, x3["fitfunc"](t), "r--")
    plt.xlabel('Time (min)')
    plt.title('Third unit Vec component in C frame (-)')

    # plt.figure(203, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    plt.figure(203, figsize=(3.5, 2.), facecolor='w', edgecolor='k')
    plt.plot(t, x[:, 4]*m2km, ".", color = colorList[1])
    plt.xlabel('Time (min)')
    plt.title('Norm in C frame (km)')



def imgProcVsExp(true, centers, radii, size):


    colorsInt = len(mpl.pyplot.get_cmap("inferno").colors)/(10.)
    colorList = []
    for i in range(10):
        colorList.append(mpl.pyplot.get_cmap("inferno").colors[int(i*colorsInt)])

    t= true[:, 0]*ns2min
    centerline = np.ones([len(t),2])*(size/2+0.5)
    found = 0
    for i in range(len(t)):
        if np.abs(centers[i,1]) <1E-10 and np.abs(centers[i,2]) < 1E-10:
            centers[i,1:3] = np.array([np.nan,np.nan])
            radii[i,1] = np.nan
        else:
            found = 1
        if found == 0:
            centerline[i,:] = np.array([np.nan,np.nan])

    # plt.figure(301, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    # # plt.figure(301, figsize=(3.5, 2.), facecolor='w', edgecolor='k')
    # plt.rcParams["font.size"] = "8"
    # plt.plot(t, true[:, 1], "+", label='Truth Xpix', color = colorList[1])
    # plt.plot(t, centerline[:,0], "--", label='X-center', color = colorList[9])
    # plt.plot(t, centers[:, 1], '.', label = "ImagProc Xpix", color = colorList[5], alpha=0.7)
    # plt.legend(loc='best')
    # try:
    #     plt.ylim([centerline[-1,1]-15, centerline[-1,1]+15])
    # except ValueError:
    #     pass
    # plt.ylabel('X (px)')
    # plt.xlabel('Time (min)')
    # #plt.savefig('Xpix.pdf')

    # plt.figure(302, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    # # plt.figure(302, figsize=(3.5, 2.), facecolor='w', edgecolor='k')
    # plt.plot(t, true[:, 2], "+", label='Truth Ypix', color = colorList[1])
    # plt.plot(t, centerline[:,1], "--", label='Y-center', color = colorList[9])
    # plt.plot(t, centers[:, 2], '.', label = "ImagProc Ypix", color = colorList[5], alpha=0.7)
    # plt.legend(loc='best')
    # try:
    #     plt.ylim([centerline[-1,1]-15, centerline[-1,1]+15])
    # except ValueError:
    #     pass
    # plt.ylabel('Y (px)')
    # plt.xlabel('Time (min)')
    # #plt.savefig('Ypix.pdf')

    fig, axes = plt.subplots(2, 1, figsize=(6, 5.5),
                             sharex=True)

    # -------------------------
    # X pixel
    # -------------------------
    ax = axes[0]

    ax.plot(
        t,
        true[:, 1],
        "+",
        label='Truth Xpix',
        color=colorList[1]
    )

    ax.plot(
        t,
        centerline[:, 0],
        "--",
        label='X-center',
        color=colorList[9]
    )

    ax.plot(
        t,
        centers[:, 1],
        ".",
        label="ImageProc Xpix",
        color=colorList[5],
        alpha=0.7
    )

    ax.set_ylabel("X (px)")
    ax.set_title("X Pixel")
    ax.legend(loc='best')
    ax.grid(True)

    # -------------------------
    # Y pixel
    # -------------------------
    ax = axes[1]

    ax.plot(
        t,
        true[:, 2],
        "+",
        label='Truth Ypix',
        color=colorList[1]
    )

    ax.plot(
        t,
        centerline[:, 1],
        "--",
        label='Y-center',
        color=colorList[9]
    )

    ax.plot(
        t,
        centers[:, 2],
        ".",
        label="ImageProc Ypix",
        color=colorList[5],
        alpha=0.7
    )

    ax.set_ylabel("Y (px)")
    ax.set_xlabel("Time (min)")
    ax.set_title("Y Pixel")
    ax.legend(loc='best')
    ax.grid(True)

    fig.suptitle("Image Processing Pixel Measurements")
    plt.tight_layout(rect=[0, 0, 1, 0.94])    

    plt.figure(312, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    # plt.figure(312, figsize=(3.5, 2.), facecolor='w', edgecolor='k')
    plt.plot(t, true[:, 3], "+", label=r'Truth $\rho$', color = colorList[1])
    plt.plot(t, radii[:, 1],'.',  label = r"ImagProc $\rho$", color = colorList[5], alpha=0.7)
    plt.title('Lunar Limb Radius: Truth vs. Image Processing')
    plt.ylabel(r'Circle Radius $\rho$ (px)')
    plt.xlabel('Time (min)')
    plt.legend(loc='best')
    plt.grid(True)
    #plt.savefig('Rhopix.pdf')


    # plt.figure(303, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    # # plt.figure(303, figsize=(3.5, 2.), facecolor='w', edgecolor='k')
    # plt.plot(t, true[:, 1] - centers[:, 1], ".", label=r'$\mathrm{X}_\mathrm{true} - \mathrm{X}_\mathrm{hough}$', color = colorList[1])
    # plt.legend(loc='best')
    # plt.ylabel('X error (px)')
    # plt.grid()
    # #plt.savefig('Xerror.pdf')

    # plt.figure(304, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    # # plt.figure(304, figsize=(3.5, 2.), facecolor='w', edgecolor='k')
    # plt.plot(t, true[:, 2] - centers[:, 2], ".", label=r'$\mathrm{Y}_\mathrm{true} - \mathrm{Y}_\mathrm{hough}$', color = colorList[1])
    # plt.legend(loc='best')
    # plt.ylabel('Y error (px)')
    # plt.grid()
    # #plt.savefig('Yerror.pdf')

    # plt.figure(305, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    # # plt.figure(305, figsize=(3.5, 2.), facecolor='w', edgecolor='k')
    # plt.plot(t, true[:, 3] - radii[:, 1], ".", label=r'$\mathrm{\rho}_\mathrm{true} - \mathrm{\rho}_\mathrm{hough}$', color = colorList[1])
    # plt.legend(loc='best')
    # plt.ylabel('Radius error (px)')
    # plt.xlabel("Time (min)")
    # plt.grid()
    fig, axes = plt.subplots(3, 1, figsize=(6, 7),
                             sharex=True)

    # -------------------------
    # X error
    # -------------------------
    ax = axes[0]

    ax.plot(
        t,
        true[:, 1] - centers[:, 1],
        ".",
        label=r'$\mathrm{X}_{\mathrm{true}}-\mathrm{X}_{\mathrm{hough}}$',
        color=colorList[1]
    )

    ax.set_ylabel("X error (px)")
    ax.set_title("X Error")
    ax.legend(loc='best')
    ax.grid(True)

    # -------------------------
    # Y error
    # -------------------------
    ax = axes[1]

    ax.plot(
        t,
        true[:, 2] - centers[:, 2],
        ".",
        label=r'$\mathrm{Y}_{\mathrm{true}}-\mathrm{Y}_{\mathrm{hough}}$',
        color=colorList[1]
    )

    ax.set_ylabel("Y error (px)")
    ax.set_title("Y Error")
    ax.legend(loc='best')
    ax.grid(True)

    # -------------------------
    # Radius error
    # -------------------------
    ax = axes[2]

    ax.plot(
        t,
        true[:, 3] - radii[:, 1],
        ".",
        label=r'$\rho_{\mathrm{true}}-\rho_{\mathrm{hough}}$',
        color=colorList[1]
    )

    ax.set_ylabel("Radius error (px)")
    ax.set_xlabel("Time (min)")
    ax.set_title("Radius Error")
    ax.legend(loc='best')
    ax.grid(True)

    fig.suptitle("Image Processing Errors")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    #plt.savefig('Rhoerror.pdf')


def plotPostFitResiduals(Res, noise):

    MeasNoise = np.zeros([len(Res[:,0]), 3])
    Noiselast = np.zeros([3,3])

    colorsInt = len(mpl.pyplot.get_cmap("inferno").colors)/(10.)
    colorList = []
    for i in range(10):
        colorList.append(mpl.pyplot.get_cmap("inferno").colors[int(i*colorsInt)])

    t= np.zeros(len(Res[:,0]))
    for i in range(len(Res[:,0])):
        t[i] = Res[i, 0]*ns2min
        if isinstance(noise, float):
            MeasNoise[i, :] = np.array([3*np.sqrt(noise),3*np.sqrt(noise),3*np.sqrt(noise)])
        else:
            if not np.isnan(noise[i, 1]):
                Noiselast = noise[i, 1:].reshape([3, 3])
                MeasNoise[i, :] = np.array([3*np.sqrt(Noiselast[0,0]), 3*np.sqrt(Noiselast[1,1]), 3*np.sqrt(Noiselast[2,2])])/5
            if np.isnan(noise[i, 1]):
                MeasNoise[i, :] = np.array([3*np.sqrt(Noiselast[0,0]), 3*np.sqrt(Noiselast[1,1]), 3*np.sqrt(Noiselast[2,2])])/5
        # Don't plot zero values, since they mean that no measurement is taken
        for j in range(len(Res[0,:])-1):
            if -1E-10 < Res[i,j+1] < 1E-10:
                Res[i, j+1] = np.nan
    if len(Res[0,:])-1 == 3:
        # plt.figure(401, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
        # plt.plot(t , Res[:, 1]*m2km, ".", label='Residual', color = colorList[1])
        # plt.plot(t , MeasNoise[:,0]*m2km, '--', label=r'Noise ($3\sigma$)', color = colorList[8])
        # plt.plot(t , -MeasNoise[:,0]*m2km, '--', color = colorList[8])
        # plt.legend(loc=2)
        # max = np.amax(MeasNoise[:,0])
        # if max >1E-15:
        #     plt.ylim([-2*max*m2km, 2*max*m2km])
        # plt.ylabel(r'$r_1$ Measured (km)')
        # plt.xlabel("Time (min)")
        # plt.grid()
        # #plt.savefig('Res1.pdf')

        # plt.figure(402, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
        # plt.plot(t , Res[:, 2]*m2km, ".",  color = colorList[1])
        # plt.plot(t , MeasNoise[:,1]*m2km, '--',  color = colorList[8])
        # plt.plot(t , -MeasNoise[:,1]*m2km, '--',  color = colorList[8])
        # max = np.amax(MeasNoise[:,1])
        # if max >1E-15:
        #     plt.ylim([-2*max*m2km, 2*max*m2km])
        # plt.ylabel(r'$r_2$ Measured (km)')
        # plt.xlabel("Time (min)")
        # plt.grid()
        # #plt.savefig('Res2.pdf')


        # plt.figure(403, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
        # plt.plot(t , Res[:, 3]*m2km, ".",  color = colorList[1])
        # plt.plot(t , MeasNoise[:,2]*m2km, '--',  color = colorList[8])
        # plt.plot(t , -MeasNoise[:,2]*m2km, '--',  color = colorList[8])
        # max = np.amax(MeasNoise[:,2])
        # if max >1E-15:
        #     plt.ylim([-2*max*m2km, 2*max*m2km])
        # plt.ylabel(r'$r_3$ Measured (km)')
        # plt.xlabel("Time (min)")
        # plt.grid()
        # #plt.savefig('Res3.pdf')
        fig, axes = plt.subplots(3, 1, figsize=(6, 7),
                                 sharex=True)

        residual_indices = [1, 2, 3]
        noise_indices = [0, 1, 2]

        for i in range(3):

            ax = axes[i]

            ax.plot(
                t,
                Res[:, residual_indices[i]] * m2km,
                ".",
                color=colorList[1],
                label="Residual"
            )

            ax.plot(
                t,
                MeasNoise[:, noise_indices[i]] * m2km,
                "--",
                color=colorList[8],
                label=r"Noise ($3\sigma$)"
            )

            ax.plot(
                t,
                -MeasNoise[:, noise_indices[i]] * m2km,
                "--",
                color=colorList[8]
            )

            max_noise = np.amax(MeasNoise[:, noise_indices[i]])

            if max_noise > 1E-15:
                ax.set_ylim([
                    -2 * max_noise * m2km,
                    2 * max_noise * m2km
                ])

            ax.set_ylabel(
                rf'$r_{i+1}$ Measured (km)'
            )

            ax.set_title(
                rf'$r_{i+1}$ Residual'
            )

            ax.legend(loc='best')
            ax.grid(True)

        axes[-1].set_xlabel("Time (min)")

        fig.suptitle("Post-Fit Position Residuals")
        plt.tight_layout(rect=[0, 0, 1, 0.94])

    if len(Res[0,:])-1 == 6:
        plt.figure(405, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
        plt.plot(t , Res[:, 1], "b.")
        plt.plot(t , MeasNoise[:,0], 'r--')
        plt.plot(t , -MeasNoise[:,0], 'r--')
        plt.legend(loc='best')
        plt.title('X-pixel component')
        plt.grid()

        plt.figure(406, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
        plt.plot(t , Res[:, 4], "b.")
        plt.plot(t , MeasNoise[:,0], 'r--')
        plt.plot(t , -MeasNoise[:,0], 'r--')
        plt.title('X-bias component')
        plt.grid()

        plt.figure(407, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
        plt.plot(t , Res[:, 2], "b.")
        plt.plot(t , MeasNoise[:,1], 'r--')
        plt.plot(t , -MeasNoise[:,1], 'r--')
        plt.title('Y-pixel component')
        plt.grid()

        plt.figure(408, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
        plt.plot(t , Res[:, 5], "b.")
        plt.plot(t , MeasNoise[:,1], 'r--')
        plt.plot(t , -MeasNoise[:,1], 'r--')
        plt.xlabel('Time (min)')
        plt.title('Y-bias component')
        plt.grid()

        plt.figure(409, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
        plt.plot(t , Res[:, 3], "b.")
        plt.plot(t , MeasNoise[:,2], 'r--')
        plt.plot(t , -MeasNoise[:,2], 'r--')
        plt.xlabel('Time (min)')
        plt.ylabel('Rho-component')
        plt.grid()

        plt.figure(410, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
        plt.plot(t , Res[:, 6], "b.")
        plt.plot(t , MeasNoise[:,2], 'r--')
        plt.plot(t , -MeasNoise[:,2], 'r--')
        plt.xlabel('Time (min)')
        plt.ylabel('Rho-bias component')
        plt.grid()

def plot_cirlces(centers, radii, validity, resolution):
    # circleIndx = []
    # for i in range(len(centers[:,0])):
    #     if validity[i, 1] == 1:
    #         circleIndx.append(i)

    # colorsInt = len(mpl.pyplot.get_cmap("inferno").colors)/(len(circleIndx)+1)
    # colorList = []
    # for i in range(len(circleIndx)):
    #     colorList.append(mpl.pyplot.get_cmap("inferno").colors[int(i*colorsInt)])

    # plt.figure(500, figsize=(3, 3), facecolor='w', edgecolor='k')
    # ax = plt.gca()
    # for i in range(len(circleIndx)):
    #     if i%30 ==0:
    #         ell_xy = Ellipse(xy=(centers[circleIndx[i],1], centers[circleIndx[i],2]),
    #                          width=radii[circleIndx[i],1], height=radii[circleIndx[i],1],
    #                          angle=0, linestyle='-', linewidth=3, color=colorList[i], alpha=0.7)
    #         ell_xy.set(facecolor='none')
    #         ax.add_patch(ell_xy)
    #         ax.invert_yaxis()
    # ax.set_xlim(0, resolution[0])
    # ax.set_ylim(resolution[1],0)
    # plt.xlabel('X-axis (px)')
    # plt.ylabel('Y-axis (px)')
    # plt.axis("equal")
    # #plt.savefig('Circles.pdf')
    circleIndx = []

    for i in range(len(centers[:, 0])):
        if validity[i, 1] == 1:
            circleIndx.append(i)

    # Only draw every 30th valid detection
    drawn_indices = [
        i for i in range(len(circleIndx))
        if i % 30 == 0
    ]

    fig = plt.figure(500, figsize=(6, 5.5), facecolor='w', edgecolor='k')
    ax = plt.gca()

    # Color progression: first -> last
    cmap = plt.get_cmap("inferno")

    for j, i in enumerate(drawn_indices):

        idx = circleIndx[i]

        if len(drawn_indices) > 1:
            color_value = j / (len(drawn_indices) - 1)
        else:
            color_value = 0.0

        ell_xy = Ellipse(
            xy=(centers[idx, 1], centers[idx, 2]),
            width=radii[idx, 1],
            height=radii[idx, 1],
            angle=0,
            linestyle='-',
            linewidth=2,
            color=cmap(color_value),
            alpha=0.7,
            fill=False
        )

        ax.add_patch(ell_xy)

    # Colorbar
    norm = mpl.colors.Normalize(
        vmin=0,
        vmax=len(drawn_indices) - 1
    )

    sm = mpl.cm.ScalarMappable(
        cmap=cmap,
        norm=norm
    )

    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Detection progression')
# Zoom to the region containing the detected circles
    x_min = np.min(centers[circleIndx, 1])
    x_max = np.max(centers[circleIndx, 1])
    y_min = np.min(centers[circleIndx, 2])
    y_max = np.max(centers[circleIndx, 2])

    # Include the circle radii
    r_max = np.max(radii[circleIndx, 1])

    margin = 1.1 * r_max

    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_max + margin, y_min - margin)

    # Axes
    # ax.set_xlim(0, resolution[0])
    # ax.set_ylim(resolution[1], 0)

    ax.set_xlabel('X pixel')
    ax.set_ylabel('Y pixel')

    ax.set_title('Detected Lunar Limb Circles')

    ax.set_aspect('equal')

    plt.tight_layout()

def plot_limb(limbPoints, numLimb,  validity, resolution):
    indx = []
    time = []
    for i in range(len(limbPoints[:,0])):
        if validity[i, 1] == 1:
            indx.append(i)

    numBerPoints = []
    colorsInt = len(mpl.pyplot.get_cmap("inferno").colors)/(len(indx)+1)
    colorList = []
    for i in range(len(indx)):
        colorList.append(mpl.pyplot.get_cmap("inferno").colors[int(i*colorsInt)])
        numBerPoints.append(numLimb[indx[i], 1])
        time.append(numLimb[indx[i], 0]*1E-9/60)

    limbX = np.zeros([len(indx), 2*2000])
    limbY = np.zeros([len(indx), 2*2000])
    for i in range(len(indx)):
        for j in range(1, int(numLimb[indx[i], 1])):
            if np.abs(limbPoints[indx[i], 2*j] + limbPoints[indx[i], 2*j-1])>1E-1:
                limbX[i,j] = limbPoints[indx[i], 2*j-1]
                limbY[i,j] =  limbPoints[indx[i], 2*j]
    plt.figure(600, figsize=(3, 3), facecolor='w', edgecolor='k')
    ax = plt.gca()
    for i in range(len(indx)):
        if i%30 ==0:
            ax.scatter(limbX[i,:int(numLimb[indx[i],1])-1], limbY[i,:int(numLimb[indx[i],1])-1], color=colorList[i], marker='.', alpha=0.2)
            ax.invert_yaxis()
    ax.set_xlim(0, resolution[0])
    ax.set_ylim(resolution[1],0)
    plt.xlabel('X-axis (px)')
    plt.ylabel('Y-axis (px)')
    plt.axis("equal")
    #plt.savefig('Limbs.pdf')

    plt.figure(601, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    plt.plot(time, numBerPoints, color=colorList[1])
    plt.xlabel('Time (min)')
    plt.ylabel('Limb Size (px)')
    #plt.savefig('LimbPoints.pdf')

class AnimatedCircles(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, size, centers, radii, validity):
        self.sensorSize = size
        self.idx = 0
        self.i = 0
        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots(num='Circles Animation')#, figsize=(10, 10), dpi=80, facecolor='w')
        self.ax.invert_yaxis()
        self.ax.set_xlim(0, self.sensorSize[0])
        self.ax.set_ylim(self.sensorSize[1],0)
        self.ax.set_aspect("equal")
        # Then setup FuncAnimation.
        self.setData(centers, radii, validity)
        self.ani = mpl.animation.FuncAnimation(self.fig, self.update, interval=100,
                                          init_func=self.setup_plot, frames=len(self.circleIndx), blit=True)
        self.ani.save('../../TestImages/circlesAnimated.gif', writer='imagemagick', fps=60)

    def setData(self, centers, radii, validity):
        self.circleIndx = []
        for i in range(len(centers[:,0])):
            if validity[i, 1] == 1:
                self.circleIndx.append(i)

        self.centers = np.zeros([len(self.circleIndx), 3])
        self.radii = np.zeros([len(self.circleIndx), 2])
        for i in range(len(self.circleIndx)):
            self.centers[i,0] = centers[self.circleIndx[i],0]
            self.centers[i, 1:] = centers[self.circleIndx[i], 1:3]
            self.radii[i,0] = centers[self.circleIndx[i],0]
            self.radii[i, 1] = radii[self.circleIndx[i], 1]
        self.colorList =  mpl.pyplot.get_cmap("inferno").colors


    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        self.scat = self.ax.scatter(self.centers[0,1], self.centers[0,2], c=self.colorList[0], s=self.radii[0,1])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self, i):
        while self.idx < len(self.circleIndx):
            xy = np.array([[self.sensorSize[0]/2+0.5, self.sensorSize[0]/2+0.5],[self.centers[i,1], self.centers[i,2]],[self.centers[i,1], self.centers[i,2]]])
            s = np.array([1, 1, self.radii[i,1]])
            c = [self.colorList[-1], self.colorList[i],self.colorList[i]]
            yield np.c_[xy[:,0], xy[:,1], s[:]], c

    def update(self, i):
        """Update the scatter plot."""
        data, color = next(self.data_stream(i))

        # Set x and y data...
        self.scat.set_offsets(data[:, :2])
        # Set sizes...
        self.scat.set_sizes((data[:, 2]/2.)**2.)
        # Set colors..
        # self.scat.set_array(data[:, 3])
        self.scat.set_edgecolor(color)
        self.scat.set_facecolor("none")
        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

def StateErrorCovarPlot(x, Pflat, FilterType, show_plots):

    nstates = int(np.sqrt(len(Pflat[0,:])-1))
    mpl.rc("figure", figsize=(3, 1.8))

    colorsInt = int(len(mpl.pyplot.get_cmap().colors)/10)
    colorList = []
    for i in range(10):
        colorList.append(mpl.pyplot.get_cmap().colors[i*colorsInt])

    P = np.zeros([len(Pflat[:,0]),nstates,nstates])
    t= np.zeros(len(Pflat[:,0]))
    for i in range(len(Pflat[:,0])):
        t[i] = x[i, 0]*ns2min
        P[i,:,:] = Pflat[i,1:37].reshape([nstates,nstates])
        for j in range(len(P[0,0,:])):
            P[i,j,j] = np.sqrt(P[i,j,j])

    for i in range(3):
        if i ==0:
            plt.figure(num=None, facecolor='w', edgecolor='k')
            plt.plot(t , x[:, i+1], color = colorList[0], label='Error')
            plt.plot(t , 3 * P[:, i, i],color = colorList[-1], linestyle = '--',  label=r'Covar (3-$\sigma$)')
            plt.plot(t , -3 * P[:, i, i], color = colorList[-1], linestyle = '--')
            plt.legend(loc='best')
            plt.ylabel('$d_' + str(i+1) + '$ Error (-)')
            plt.ylim([-0.1,0.1])
            plt.xlabel('Time (min)')
            simHelpers.saveFigurePDF('StateCovarHeading'+ FilterType + str(i) , plt, './')
            if show_plots:
                plt.show()
            plt.close("all")
        else:
            plt.figure(num=None, facecolor='w', edgecolor='k')
            plt.plot(t , x[:, i+1], color = colorList[0])
            plt.plot(t , 3 * P[:, i, i],color = colorList[-1], linestyle = '--')
            plt.plot(t , -3 * P[:, i, i], color = colorList[-1], linestyle = '--')
            plt.ylabel('$d_' + str(i+1) + '$ Error (-)')
            plt.xlabel('Time (min)')
            plt.ylim([-0.1,0.1])
            simHelpers.saveFigurePDF('StateCovarHeading'+ FilterType + str(i) , plt, './')
            if show_plots:
                plt.show()
            plt.close("all")

    if nstates>3:
        for i in range(3,nstates):
            if i == 3:
                plt.figure(num=None, facecolor='w', edgecolor='k')
                plt.plot(t, x[:, i + 1], color = colorList[0], label='Error')
                plt.plot(t, 3 * P[:, i, i], color = colorList[-1],linestyle = '--', label=r'Covar (3-$\sigma$)')
                plt.plot(t, -3 * P[:, i, i], color = colorList[-1],linestyle = '--')
                plt.ylim([-0.0007,0.0007])
                # plt.legend(loc='best')
                if nstates == 5:
                    plt.ylabel(r'$\omega_' + str(i - 1) + r'$ Error (-)')
                else:
                    plt.ylabel(r'$d\'_' + str(i-2) + r'$ Error (-)')
                plt.xlabel('Time (min)')
                simHelpers.saveFigurePDF('StateCovarRate' + FilterType + str(i), plt, './')
                if show_plots:
                    plt.show()
                plt.close("all")
            else:
                plt.figure(num=None, facecolor='w', edgecolor='k')
                plt.plot(t, x[:, i + 1], color = colorList[0])
                plt.plot(t, 3 * P[:, i, i], color = colorList[-1],linestyle = '--')
                plt.plot(t, -3 * P[:, i, i], color = colorList[-1],linestyle = '--')
                plt.ylim([-0.0007,0.0007])
                if nstates == 5:
                    plt.ylabel(r'$\omega_' + str(i - 1) + r'$ Error (-)')
                else:
                    plt.ylabel(r'$d\'_' + str(i-2) + r'$ Error (-)')
                plt.xlabel('Time (min)')
                simHelpers.saveFigurePDF('StateCovarRate' + FilterType + str(i), plt, './')
                if show_plots:
                    plt.show()
                plt.close("all")
    plt.close("all")

def PostFitResiduals(Res, covar_B, FilterType, show_plots):

    mpl.rc("figure", figsize=(3, 1.8))

    colorsInt = int(len(mpl.pyplot.get_cmap().colors) / 10)
    colorList = []
    for i in range(10):
        colorList.append(mpl.pyplot.get_cmap().colors[i * colorsInt])

    MeasNoise = np.zeros([len(Res[:, 0]), 3])
    prevNoise = np.zeros(3)
    t = np.zeros(len(Res[:, 0]))
    constantVal = np.array([np.nan] * 4)
    for i in range(len(Res[:, 0])):
        t[i] = Res[i, 0] * ns2min
        if np.abs(covar_B[i, 1]) < 1E-15 and prevNoise[0] == 0:
            MeasNoise[i,:] = np.full(3, np.nan)
        elif np.abs(covar_B[i, 1]) < 1E-15 and prevNoise[0] != 0:
            MeasNoise[i,:] = prevNoise
        else:
            MeasNoise[i,0] = 3 * np.sqrt(covar_B[i, 1])
            MeasNoise[i,1] = 3 * np.sqrt(covar_B[i, 1 + 4])
            MeasNoise[i,2] = 3 * np.sqrt(covar_B[i, 1 + 8])
            prevNoise = MeasNoise[i,:]

       # Don't plot constant values, they mean no measurement is taken
        if i > 0:
            for j in range(1, 4):
                constantRes = np.abs(Res[i, j] - Res[i - 1, j])
                if constantRes < 1E-10 or np.abs(constantVal[j - 1] - Res[i, j]) < 1E-10:
                    constantVal[j - 1] = Res[i, j]
                    Res[i, j] = np.nan

    for i in range(3):
        if i == 0:
            plt.figure(num=None, dpi=80, facecolor='w', edgecolor='k')
            plt.plot(t, Res[:, i + 1], color=colorList[0], linestyle='', marker='.', label='Residual')
            plt.plot(t, MeasNoise[:,i], color=colorList[-1], linestyle="--", label=r'Noise (3-$\sigma$)')
            plt.plot(t, -MeasNoise[:,i], color=colorList[-1], linestyle="--", )
            plt.legend(loc='best')
            plt.ylabel('$r_' + str(i + 1) + '$ (-)')
            plt.xlabel('Time (min)')
            plt.ylim([-2*MeasNoise[-1,i], 2*MeasNoise[-1,i]])
            simHelpers.saveFigurePDF('PostFit' + FilterType + str(i), plt, './')
            if show_plots:
                plt.show()
            plt.close("all")
        else:
            plt.figure(num=None, dpi=80, facecolor='w', edgecolor='k')
            plt.plot(t, Res[:, i + 1], color=colorList[0], linestyle='', marker='.')
            plt.plot(t, MeasNoise[:,i], color=colorList[-1], linestyle="--")
            plt.plot(t, -MeasNoise[:,i], color=colorList[-1], linestyle="--", )
            plt.ylabel('$r_' + str(i + 1) + '$ (-)')
            plt.xlabel('Time min')
            plt.ylim([-2*MeasNoise[-1,i], 2*MeasNoise[-1,i]])
            simHelpers.saveFigurePDF('PostFit' + FilterType + str(i), plt, './')
            if show_plots:
                plt.show()
            plt.close("all")

    plt.close("all")

class AnimatedLimb(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, size, limb, centers, validity):
        self.stream = self.data_stream()
        self.sensorSize = size
        self.idx = 0
        self.i = 0
        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots(num='Limb Animation')  # , figsize=(10, 10), dpi=80, facecolor='w')
        self.ax.invert_yaxis()
        self.ax.set_xlim(0, self.sensorSize[0])
        self.ax.set_ylim(self.sensorSize[1], 0)
        self.ax.set_aspect("equal")
        # Then setup FuncAnimation.
        self.setData(limb, validity)
        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = mpl.animation.FuncAnimation(self.fig, self.update, interval=100,
                                          init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        self.scat = self.ax.scatter(self.centers[0,1], self.centers[0,2], c=self.colorList[0], s=self.radii[0,1])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,


    def setData(self, centers, limbPoints, validity):
        self.pointIndx = []
        for i in range(len(limbPoints[:,0])):
            if validity[i, 1] == 1:
                self.pointIndx.append(i)
        self.centers = np.zeros([len(self.pointIndx), 2])
        self.points = np.zeros([len(self.pointIndx), 2*1000])
        for i in range(len(self.pointIndx)):
            self.centers[i, 1:] = centers[self.pointIndx[i], 1:2]
            self.points[i,0] = limbPoints[self.pointIndx[i],1:]
        self.colorList =  mpl.pyplot.get_cmap("inferno").colors


    def data_stream(self, i):
        """Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect."""
        limb = []
        while self.idx < len(self.circleIndx):
            xy = np.array([[self.sensorSize[0]/2+0.5, self.sensorSize[0]/2+0.5],[self.centers[i,1], self.centers[i,2]]])
            for j in range(len(self.points[i,:])/2):
                if self.points[i,j] >1E-1 or self.points[i,j+1] >1E-1:
                    limb.append([self.points[i,j], self.points[i,j+1]])
            c = [self.colorList[-1], self.colorList[i], self.colorList[i]]
            yield np.c_[xy[:,0], xy[:,1], limb[:,0], limb[:,1]], c

    def update(self, i):
        """Update the scatter plot."""
        data, color = next(self.data_stream(i))

        # Set x and y data...
        self.scat.set_offsets(data[:, :2])
        # Set sizes...
        self.scat.set_offsets(data[:, 2:4])
        # Set colors..
        # self.scat.set_array(data[:, 3])
        self.scat.set_edgecolor(color)
        self.scat.set_facecolor("none")
        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat
