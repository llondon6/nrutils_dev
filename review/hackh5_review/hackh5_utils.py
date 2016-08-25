#!/usr/bin/python

# This script incorporates code from https://github.com/ligo-cbc/pycbc
# covered by the following terms:
#
# > Copyright (C) 2015 Christopher M. Biwer
# >
# > This program is free software; you can redistribute it and/or modify it
# > under the terms of the GNU General Public License as published by the
# > Free Software Foundation; either version 3 of the License, or (at your
# > option) any later version.
# >
# > This program is distributed in the hope that it will be useful, but
# > WITHOUT ANY WARRANTY; without even the implied warranty of
# > MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# > Public License for more details.
# >
# > You should have received a copy of the GNU General Public License along
# > with this program; if not, write to the Free Software Foundation, Inc.,
# > 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import h5py                     as h5
import matplotlib.pyplot        as plt
import numpy                    as np
import xml.etree.ElementTree    as xml
from pycbc                      import pnutils
from pycbc.detector             import Detector
from pycbc.inject               import legacy_approximant_name
from pycbc.waveform             import get_td_waveform
from matplotlib.ticker          import AutoMinorLocator
from scipy.interpolate          import InterpolatedUnivariateSpline as IUS

# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (150, 150, 150), (199, 199, 199), # (127,127,127)
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

# map order integer to a string that can be parsed by lalsimulation
pn_orders = {
    'default'          : -1,
    'zeroPN'           : 0,
    'onePN'            : 2,
    'onePointFivePN'   : 3,
    'twoPN'            : 4,
    'twoPointFivePN'   : 5,
    'threePN'          : 6,
    'threePointFivePN' : 7,
    'pseudoFourPN'     : 8,
}

def SimInspiralNRWaveformGetRotationAnglesFromH5File(h5File,inclination,orb_phase):
    """
    Compute the angles necessary to rotate from the intrinsic NR source frame
    into the LAL frame. See DCC-T1600045 for details.

    Implementation copied from
    https://github.com/lscsoft/lalsuite/blob/d6e071988b46ec4497ac0f8fe3fe4f9d91e349d5/lalsimulation/src/LALSimIMRNRWaveforms.c#L156

    Parameters
    ----------
    h5File: hdf5.File
        h5 file containing the NR data. See DCC-T1500606 for details
    inclination: float
        Inclination of source
    orb_phase: float
        Orbital reference phase

    Returns
    -------
    theta: float
        Inclination angle of source
    psi: float
        Azimuth angle of source
    calpha: float
        Cosine of the polarization angle
    salpha: float
        Sine of the polarization angle
    z_wave: np.Array
        Vector pointing in the target direction of propogation of the
        gravitational wave.
    """

    ln_hat_x = h5File.attrs['LNhatx']
    ln_hat_y = h5File.attrs['LNhaty']
    ln_hat_z = h5File.attrs['LNhatz']

    ln_hat_norm = np.sqrt(ln_hat_x*ln_hat_x + ln_hat_y*ln_hat_y + ln_hat_z*ln_hat_z)

    ln_hat_x = ln_hat_x / ln_hat_norm
    ln_hat_y = ln_hat_y / ln_hat_norm
    ln_hat_z = ln_hat_z / ln_hat_norm

    n_hat_x = h5File.attrs['nhatx']
    n_hat_y = h5File.attrs['nhaty']
    n_hat_z = h5File.attrs['nhatz']

    n_hat_norm = np.sqrt(n_hat_x*n_hat_x + n_hat_y*n_hat_y + n_hat_z*n_hat_z)

    n_hat_x = n_hat_x / n_hat_norm
    n_hat_y = n_hat_y / n_hat_norm
    n_hat_z = n_hat_z / n_hat_norm


    corb_phase = np.cos(orb_phase)
    sorb_phase = np.sin(orb_phase)
    sinclination = np.sin(inclination)
    cinclination = np.cos(inclination)

    ln_cross_n_x = ln_hat_y*n_hat_z - ln_hat_z*n_hat_y
    ln_cross_n_y = ln_hat_z*n_hat_x - ln_hat_x*n_hat_z
    ln_cross_n_z = ln_hat_x*n_hat_y - ln_hat_y*n_hat_x

    z_wave_x = sinclination*(sorb_phase*n_hat_x + corb_phase*ln_cross_n_x)
    z_wave_y = sinclination*(sorb_phase*n_hat_y + corb_phase*ln_cross_n_y)
    z_wave_z = sinclination*(sorb_phase*n_hat_z + corb_phase*ln_cross_n_z)

    z_wave_x += cinclination*ln_hat_x
    z_wave_y += cinclination*ln_hat_y
    z_wave_z += cinclination*ln_hat_z

    theta = np.arccos(z_wave_z)

    if np.abs(z_wave_z - 1.0 ) < 0.000001:
        psi = 0.5
    else:
        if np.abs(z_wave_x / np.sin(theta)) > 1.0 :
            if np.abs(z_wave_x / np.sin(theta)) < 1.00001 :
                psi = np.pi if (z_wave_x*np.sin(theta)) < 0.0 else 0.0
        else:
            psi = np.arccos(z_wave_x / np.sin(theta))
        if z_wave_y < 0.0 :
            psi = 2*np.pi - psi

    stheta = np.sin(theta)
    ctheta = np.cos(theta)
    spsi = np.sin(psi)
    cpsi = np.cos(psi)

    theta_hat_x = cpsi*ctheta
    theta_hat_y = spsi*ctheta
    theta_hat_z = - stheta

    psi_hat_x = -spsi
    psi_hat_y = cpsi
    psi_hat_z = 0.0

    n_dot_theta             = n_hat_x*theta_hat_x + n_hat_y*theta_hat_y + n_hat_z*theta_hat_z
    ln_cross_n_dot_theta    = ln_cross_n_x*theta_hat_x + ln_cross_n_y*theta_hat_y + ln_cross_n_z*theta_hat_z
    n_dot_psi               = n_hat_x*psi_hat_x + n_hat_y*psi_hat_y + n_hat_z*psi_hat_z
    ln_cross_n_dot_psi      = ln_cross_n_x*psi_hat_x + ln_cross_n_y*psi_hat_y + ln_cross_n_z*psi_hat_z

    salpha = corb_phase*n_dot_theta - sorb_phase*ln_cross_n_dot_theta
    calpha = corb_phase*n_dot_psi - sorb_phase*ln_cross_n_dot_psi

    return theta, psi, calpha, salpha, np.array((z_wave_x, z_wave_y, z_wave_z))

def SimInspiralNRWaveformGetCorrectedRotationAnglesFromH5File(
    h5File,h5FileSource,inclinationSource,orb_phaseSource,polarizationSource):
    """
    Compoute the corrected angles for a hacked h5File for an NR injection

    Given a hacked NR file calculated the (corrected) angles nessesary to
    produce a detector response equal to that of a non-hacked NR file with the
    parameters given in the interface.

    Parameters
    ----------
    h5File: hdf5.File
        h5 file containing the NR data. See DCC-T1500606 for details. This file
        has hacked `Ln_hat`, `n_hat` and spin values to make it work with the
        NR injection infrascturcture.
    h5FileSource: hdf5.File
        h5 file containing the same NR data as h5File. The metadata in this
        file is correct according to the specification described in
        DCC-T1500606.
    inclinationSource: float
        Desired inclination of source in h5FileSource
    orb_phaseSource: float
        Desired orbital reference phase in h5FileSource
    polarizationSource: float
        Desired detector polarization for source in h5FileSource

    Returns
    -------
    inclination: float
        inclination angle of source in h5File that will produce the same
        detector response as inclinationSource for h5FileSource when combined
        with the other returned parameters.
    orb_phase: float
        Orbital phase of source in h5File that will produce the same
        detector response as orb_phaseSource for h5FileSource when combined
        with the other returned parameters.
    polarization: float
        Polarization angle of source in h5File that will produce the same
        detector response as polarizationSource for h5FileSource when combined
        with the other returned parameters.
    """

    theta, psi, calpha, salpha, z_wave = SimInspiralNRWaveformGetRotationAnglesFromH5File(
        h5FileSource,inclinationSource,orb_phaseSource)

    ln_hat_x = h5File.attrs['LNhatx']
    ln_hat_y = h5File.attrs['LNhaty']
    ln_hat_z = h5File.attrs['LNhatz']

    ln_hat_norm = np.sqrt(ln_hat_x*ln_hat_x + ln_hat_y*ln_hat_y + ln_hat_z*ln_hat_z)

    ln_hat_x = ln_hat_x / ln_hat_norm
    ln_hat_y = ln_hat_y / ln_hat_norm
    ln_hat_z = ln_hat_z / ln_hat_norm

    n_hat_x = h5File.attrs['nhatx']
    n_hat_y = h5File.attrs['nhaty']
    n_hat_z = h5File.attrs['nhatz']

    n_hat_norm = np.sqrt(n_hat_x*n_hat_x + n_hat_y*n_hat_y + n_hat_z*n_hat_z)

    n_hat_x = n_hat_x / n_hat_norm
    n_hat_y = n_hat_y / n_hat_norm
    n_hat_z = n_hat_z / n_hat_norm

    ln_cross_n_x = ln_hat_y*n_hat_z - ln_hat_z*n_hat_y
    ln_cross_n_y = ln_hat_z*n_hat_x - ln_hat_x*n_hat_z
    ln_cross_n_z = ln_hat_x*n_hat_y - ln_hat_y*n_hat_x

    # Solve for trig terms
    v = np.linalg.solve(
        np.array((
            (n_hat_x, ln_cross_n_x, ln_hat_x),
            (n_hat_y, ln_cross_n_y, ln_hat_y),
            (n_hat_z, ln_cross_n_z, ln_hat_z))),
        z_wave)

    # Two solutions exist for the *corrected* angles, chose smallest inc diff
    inc1        = np.arccos(v[2])
    inc2        = 2*np.pi-inc1
    incS        = inclinationSource%(2*np.pi)
    inclination = inc1 if np.abs(inc1-incS)<np.abs(inc2-incS) else inc2

    # Based on inclination choice solve for oribital phase correction
    orb_phase   = np.arctan2(v[0]/np.sin(inclination),v[1]/np.sin(inclination))%(2*np.pi)

    # Determine source alpha
    alphaSource = np.arctan2(salpha,calpha)

    # Caclulate hacked alpha
    theta, psi, calpha, salpha, z_wave = SimInspiralNRWaveformGetRotationAnglesFromH5File(
        h5File,inclination,orb_phase)

    # Calculate alpha correction
    alphaCor = alphaSource - np.arctan2(salpha,calpha)

    # Calculate corrected polarization
    polarization = (polarizationSource+alphaCor)%(2*np.pi)

    return inclination, orb_phase, polarization

def get_td_waveform_resp(params):
    """
    Generate time domain data of gw detector response

    This function will produce data of a gw detector response based on a
    numerical relativity waveform.

    Parameters
    ----------
    params: object
        The fields of this object correspond to the kwargs of the
        `pycbc.waveform.get_td_waveform()` method and the positional
        arguments of `pycbc.detector.Detector.antenna_pattern()`. For the later
        the fields should be supplied as `params.ra`, `.dec`, `.polarization`
        and `.geocentric_end_time`

    Returns
    -------
    h_plus: pycbc.Types.TimeSeries
    h_cross: pycbc.Types.TimeSeries
    pats: dictionary
        Dictionary containing 'H1' and 'L1' keys. Each key maps to an object
        of containing the field `.f_plus` and `.f_cross` corresponding to
        the plus and cross antenna patterns for the two ifos 'H1' and 'L1'.
    """

    # # construct waveform string that can be parsed by lalsimulation
    waveform_string = params.approximant
    if not pn_orders[params.order] == -1:
        waveform_string += params.order
    name, phase_order = legacy_approximant_name(waveform_string)

    # Populate additional fields of params object
    params.mchirp, params.eta = pnutils.mass1_mass2_to_mchirp_eta(
        params.mass1, params.mass2)
    params.waveform           = waveform_string
    params.approximant        = name
    params.phase_order        = phase_order

    # generate waveform
    h_plus, h_cross = get_td_waveform(params)

    # Generate antenna patterns for all ifos
    pats = {}
    for ifo in params.instruments:

        # get Detector instance for IFO
        det = Detector(ifo)

        # get antenna pattern
        f_plus, f_cross = det.antenna_pattern(
            params.ra,
            params.dec,
            params.polarization,
            params.geocentric_end_time)

        # Populate antenna patterns with new pattern
        pat         = type('AntennaPattern', (object,), {})
        pat.f_plus  = f_plus
        pat.f_cross = f_cross
        pats[ifo]   = pat

    return h_plus, h_cross, pats

def plot_td_waveform_resp(
    params1, params2, ant=True, outPath='td-detector-res.pdf', resi=True, norm=True, xmin=None, xmax=None):
    """
    Generate time domain plot of gw detector response

    This function will produce a two column, two row plot of a two gw detector
    responses based on a numerical relativity waveform. In each column the LHO
    and LLO detector responses are plotted for the two cases. The individual
    polarizations of the strain are also plotted as background.

    This figure is designed to be ploted to compare polarizations and so the
    titles of the columns display the polarizations, however there is no
    restriction on the two parameter objects.

    Parameters
    ----------
    params1: object
        The parameters of the waveform to plot in the left column. The fields of
        this object correspond to the kwargs of the
        `pycbc.waveform.get_td_waveform()` method and the positional arguments
        of `pycbc.detector.Detector.antenna_pattern()`. For the later the fields
        should be supplied as `params.ra`, `.dec`, `.polarization` and
        `.geocentric_end_time`.
    params2: object
        The parameters of the waveform to plot in the right column. The fields
        of this object correspond to the kwargs of the
        `pycbc.waveform.get_td_waveform()` method and the positional arguments
        of `pycbc.detector.Detector.antenna_pattern()`. For the later the fields
        should be supplied as `params.ra`, `.dec`, `.polarization` and
        `.geocentric_end_time`.
    ant: boolean, optional
        If True, plot Fp*hp and Fc*hc in the background. If False plot hp and
        hc in the background.
    outPath: string, optional
        Image file path to which the plot can be written.
    resi: boolean, optional
        Plot the residuals of the two waveforms.
    xmin: float, optional
        Minimum x-axis limit.
    xmax: float, optional
        Maximum x-axis limit.
    """

    # Custom configuration
    with plt.rc_context(dict({},**{
        'legend.fontsize':10,
        'axes.labelsize':11,
        'font.family':'serif',
        'font.size':11,
        'xtick.labelsize':11,
        'ytick.labelsize':11,
        'figure.figsize':(16,10),
        'savefig.dpi':80,
        'figure.subplot.bottom': 0.06,
        'figure.subplot.left': 0.06,
        'figure.subplot.right': 0.975,
        'figure.subplot.top': 0.975,
        'axes.unicode_minus': False
    })):

        # Prepare figure
        fig       = plt.figure()
        gs_strain = plt.GridSpec(10, 2, hspace=0, wspace=0)

        # Add axes
        ax1H1 = fig.add_subplot(gs_strain[0:5,0:2])
        ax1L1 = fig.add_subplot(gs_strain[5:10,0:2],sharex=ax1H1,sharey=ax1H1)

        # Generate data
        hp1, hc1, ps1 = get_td_waveform_resp(params1)
        hp2, hc2, ps2 = get_td_waveform_resp(params2)

        colors1 = [tableau20[0],tableau20[1]]
        colors2 = [tableau20[6],tableau20[7]]

        if resi:

            # Plot H1 response
            hResp1H1 = -1*(ps1['H1'].f_plus*hp1+ps1['H1'].f_cross*hc1)
            hResp2H1 = -1*(ps2['H1'].f_plus*hp2+ps2['H1'].f_cross*hc2)
            t0       = np.max((hp1.sample_times[0],hp2.sample_times[0]))
            t1       = np.min((hp1.sample_times[-1],hp2.sample_times[-1]))
            t        = np.linspace(t0,t1,len(hp1))
            hResp1H1 = IUS(hp1.sample_times,hResp1H1,k=5)
            hResp2H1 = IUS(hp2.sample_times,hResp2H1,k=5)
            hRespH1  = hResp2H1(t)-hResp1H1(t)
            hRespH1  /= hResp1H1(t)/100.0 if norm else 1
            ax1H1.plot(
                t,
                hRespH1 if not norm else np.abs(hRespH1),
                color=colors1[0],
                lw=1.5)

            # Plot L1 response
            hResp1L1 = -1*(ps1['L1'].f_plus*hp1+ps1['L1'].f_cross*hc1)
            hResp2L1 = -1*(ps2['L1'].f_plus*hp2+ps2['L1'].f_cross*hc2)
            t0       = np.max((hp1.sample_times[0],hp2.sample_times[0]))
            t1       = np.min((hp1.sample_times[-1],hp2.sample_times[-1]))
            t        = np.linspace(t0,t1,len(hp1))
            hResp1L1 = IUS(hp1.sample_times,hResp1L1,k=5)
            hResp2L1 = IUS(hp2.sample_times,hResp2L1,k=5)
            hRespL1  = hResp2L1(t)-hResp1L1(t)
            hResp1L1 = hResp1L1(t)
            mask     = hResp1L1 == 0
            hResp1L1[mask] = 1
            hRespL1  /= hResp1L1/100.0 if norm else 1
            ax1L1.plot(
                t,
                hRespL1 if not norm else np.abs(hRespL1),
                color=colors1[0],
                lw=1.5)

            # Prepare H1 gridlines
            ax1H1.grid(alpha=0.3)
            ax1H1.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax1H1.grid(False)

            # Prepare L1 gridlines
            ax1L1.grid(alpha=0.3)
            ax1L1.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax1L1.grid(False)

            # Prepare strain range
            ymax = np.max((np.mean(np.abs(hRespH1)),np.mean(np.abs(hRespH1))))*2 if norm else np.max((np.abs(hRespH1).max(),np.abs(hRespL1).max()))

            ymax *= 1.25
            ax1H1.set_ylim(0 if norm else -ymax,ymax)

        else:

            # Plot left and right columns
            columns = (
                (ax1H1,ax1L1,hp1,hc1,ps1,params1,colors1),
                (ax1H1,ax1L1,hp2,hc2,ps2,params2,colors2))

            for axH1,axL1,hp,hc,ps,params,colors in columns:

                # Plot h+
                axH1.plot(
                    hp.get_sample_times(),
                    ps['H1'].f_plus*hp if ant else hp,
                    color=colors[1],
                    ls='-',
                    label='$F_+h_+$' if ant else '$h_+$')

                # Plot hx
                axH1.plot(
                    hc.get_sample_times(),
                    ps['H1'].f_cross*hc if ant else hc,
                    color=colors[1],
                    ls='--',dashes=(2,2),
                    label='$F_{\\times}h_{\\times}$' if ant else '$h_{\\times}$')

                # Plot H1 response
                axH1.plot(
                    hc.get_sample_times(),
                    -1*(ps['H1'].f_plus*hp+ps['H1'].f_cross*hc),
                    color=colors[0],
                    label='$\\psi={0:.0f}^{{\circ}}$'.format((180.0/np.pi)*params.polarization),
                    lw=1.5)

                # Plot h+
                axL1.plot(
                    hp.get_sample_times(),
                    ps['L1'].f_plus*hp if ant else hp,
                    color=colors[1],
                    ls='-',
                    label='$F_+h_+$' if ant else '$h_+$')

                # Plot hx
                axL1.plot(
                    hc.get_sample_times(),
                    ps['L1'].f_cross*hc if ant else hc,
                    color=colors[1],
                    ls='--',dashes=(2,2),
                    label='$F_{\\times}h_{\\times}$' if ant else '$h_{\\times}$')

                # Plot L1 response
                axL1.plot(
                    hc.get_sample_times(),
                    ps['L1'].f_plus*hp+ps['L1'].f_cross*hc,
                    color=colors[0],
                    label='$\\psi={0:.0f}^{{\circ}}$'.format((180.0/np.pi)*params.polarization),
                    lw=1.5)

                # Prepare H1 gridlines
                axH1.grid(alpha=0.3)
                axH1.xaxis.set_minor_locator(AutoMinorLocator(2))
                axH1.grid(False)

                # Prepare L1 gridlines
                axL1.grid(alpha=0.3)
                axL1.xaxis.set_minor_locator(AutoMinorLocator(2))
                axL1.grid(False)

            # Prepare strain range
            if ant:
                ymax = np.max((
                    np.max(np.abs(ps1['H1'].f_plus*hp1.data)),
                    np.max(np.abs(ps1['H1'].f_cross*hc1.data)),
                    np.max(np.abs(ps1['L1'].f_plus*hp1.data)),
                    np.max(np.abs(ps1['L1'].f_cross*hc1.data)),
                    np.max(np.abs(ps2['H1'].f_plus*hp2.data)),
                    np.max(np.abs(ps2['H1'].f_cross*hc2.data)),
                    np.max(np.abs(ps2['L1'].f_plus*hp2.data)),
                    np.max(np.abs(ps2['L1'].f_cross*hc2.data))))
            else:
                ymax = np.max((
                    np.max(np.abs(hp1.data)),
                    np.max(np.abs(hc1.data)),
                    np.max(np.abs(hp2.data)),
                    np.max(np.abs(hc2.data))))

            ymax *= 1.25
            ax1H1.set_ylim(-ymax,ymax)

        # Prepare time range
        xmin = (hp1.get_sample_times()[0]) if xmin == None else xmin
        xmax = (hp1.get_sample_times()[-1])/5 if xmax == None else xmax
        ax1H1.set_xlim(xmin,xmax)

        # Condition yticks
        tickTol = ax1H1.get_ylim()[1]
        ticks   = ax1H1.get_yticks()
        ticks   = [e for e in ticks if (np.abs(e)-(4./5.)*tickTol) < 0]
        ax1H1.set_yticks(ticks)
        ax1L1.set_yticks(ticks)

        # Prepare Strain label
        yH1      = ax1H1.get_yaxis()
        offset   = yH1.major.formatter \
            .format_data_short(ax1H1.get_yticks()[-1]).split('e')
        exponent = '0' if len(offset)==1 else offset[1].rstrip()
        ax1H1.set_ylabel(
            "Residual (%)" if norm else "Strain $\mathregular{{(10^{{{0:s}}})}}$".format(exponent),x=0,y=0)

        # Prepare Time label
        ax1L1.set_xlabel('Time (s)',x=0.5,y=0)

        # Prepare polarization labels
        titleStr = (
            '$\\iota_{{1}}={0:.2f}^{{\circ}}$\t$\\iota_{{2}}={1:.2f}^{{\circ}}$\n'
            '$\\psi_{{1}}={2:.2f}^{{\circ}}$\t$\\psi_{{2}}={3:.2f}^{{\circ}}$\n'
            '$\\phi_{{1}}={4:.2f}^{{\circ}}$\t\t$\\phi_{{2}}={5:.2f}^{{\circ}}$')
        iota1 = (180.0/np.pi)*params1.inclination
        iota2  = (180.0/np.pi)*params2.inclination
        psi1 = (180.0/np.pi)*params1.polarization
        psi2  = (180.0/np.pi)*params2.polarization
        phi1 = (180.0/np.pi)*params1.coa_phase
        phi2  = (180.0/np.pi)*params2.coa_phase

        ax1H1Pos = ax1H1.get_position()
        fig.text(
            ax1H1Pos.x1-0.01,
            ax1H1Pos.y1-0.01,
            titleStr.format(iota1,iota2,psi1,psi2,phi1,phi2),
            va="top",
            ha="right")

        # Remove offset text from yaxes
        for ax in (ax1H1,ax1L1):
            ax.get_yaxis().get_offset_text().set_visible(False)

        # Remove ticks from shared axes
        plt.setp(ax1H1.get_xticklabels(), visible=False)

        # Dump plot
        fig.savefig(outPath)

        # Kill figure
        plt.show()
        plt.close(fig)
