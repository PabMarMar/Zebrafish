import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import interpolate
from scipy.optimize import curve_fit
import sympy as sp
from scipy import stats

plt.rcParams["font.family"] = "serif"



plt.rcParams["mathtext.fontset"] = "dejavuserif"

def graph(signal):
    path = "G:\\Mi unidad\\Diego\\Drogas\\Ciclo_final_summaries"
    files = os.listdir(path)
    select = []
    for i in files:
        if i[0:7] == 'Summary':
            select.append(i)
        else:
            continue
    result = pd.DataFrame(np.zeros((len(select), 5)))

    for i in range(len(select)):
        data = pd.read_table(path+'\\'+select[i], delimiter='	')
        if select[i] == "Summary_44_1_R3.txt":
            data.iloc[:, len(data.columns)-7]/=3.2055
            print('1')
        if select[i] == "Summary_44_2_R3.txt":
            data.iloc[:, len(data.columns)-7]/=3.2055*3.2055
            print('2')
        if select[i] == "Summary_44_3_R3.txt":
            data.iloc[:, len(data.columns)-7]/=3.2055*3.2055
            print('3')
        if select[i] == "Summary_44_4_R3.txt":
            data.iloc[:, len(data.columns)-7]/=3.2055*3.2055
            print('4')
        if select[i] == "Summary_44_5_R3.txt":
            data.iloc[:, len(data.columns)-7]/=3.2055*3.2055
            print('5')
        if select[i] == "Summary_44_6_R3.txt":
            data.iloc[:, len(data.columns)-7]/=3.2055*3.2055
            print('6')

        nrow = len(data)
        diff = data.iloc[:, len(data.columns)-1]
        prog = data.iloc[:, len(data.columns)-2]
        vol = data.iloc[:, len(data.columns)-7]
        mask = (prog < diff)
        if all(diff == 0):
            continue
        #print(len(diff[mask]))
        #print(len(diff[mask][diff[mask]/vol[mask]>.25]))
        D = len(diff[mask][diff[mask]/vol[mask]>.25])

        P = nrow - D
        false = sum(mask*(prog == 0))
        true = sum(mask*(prog != 0))
        result.iloc[i, :] = [select[i], D, P, false, true]

    # result.to_csv(path + '\\Results\\' + 'try.csv', sep=';')
    result.columns = ['Name', 'Dif', 'Prog', 'False', 'True']
    #results = pd.DataFrame(data = result, columns = ['Name', 'Dif', 'Prog','False','True'])
    #print(result['True'])
    #print(result['False'])
    #print(result['Dif'])
    #print(result)
    #np.savetxt('data.txt',result, fmt = "%s")
    result['True'] = result['True']/result['False']
    result = result[result['True'] > 0]
    #print(result['True'])
    #print(result['Dif'])
    result['Name'] = result['Name'].str.replace('Summary_', '')
    result['Name'] = result['Name'].str[0:2]

    #--------------------------------------------------

    P = result['Prog']
    D = result['Dif']
    T = result['Name']

    P = np.array(P)
    T = np.array(T).astype(int)
    D = np.array(D)

    prog, dif, tot, progerr, diferr, toterr, tt = reset_means(P, D, T)


    path = "G:\\Mi unidad\\Diego\\Drogas\\XAV_final_summaries"
    files = os.listdir(path)
    select = []
    for i in files:
        if i[0:7] == 'Summary':
            select.append(i)
        else:
            continue
    result = pd.DataFrame(np.zeros((len(select), 5)))

    for i in range(len(select)):
        data = pd.read_table(path+'\\'+select[i], delimiter='	')
        nrow = len(data)
        diff = data.iloc[:, len(data.columns)-1]
        prog = data.iloc[:, len(data.columns)-2]
        vol = data.iloc[:, len(data.columns)-7]
        mask = (prog < diff)
        if all(diff == 0):
            continue
        #print(len(diff[mask]))
        #print(len(diff[mask][diff[mask]/vol[mask]>.25]))
        DX = len(diff[mask][diff[mask]/vol[mask]>.25])

        PX = nrow - DX
        false = sum(mask*(prog == 0))
        true = sum(mask*(prog != 0))
        result.iloc[i, :] = [select[i], DX, PX, false, true]

    # result.to_csv(path + '\\Results\\' + 'try.csv', sep=';')
    result.columns = ['Name', 'Dif', 'Prog', 'False', 'True']
    #results = pd.DataFrame(data = result, columns = ['Name', 'Dif', 'Prog','False','True'])
    #print(result['True'])
    #print(result['False'])
    #print(result['Dif'])
    #print(result)
    #np.savetxt('data.txt',result, fmt = "%s")
    result['True'] = result['True']/result['False']
    result = result[result['True'] > 0]
    #print(result['True'])
    #print(result['Dif'])
    result['Name'] = result['Name'].str.replace('Summary_', '')
    result['Name'] = result['Name'].str[0:2]

    #--------------------------------------------------

    PX = result['Prog']
    DX = result['Dif']
    TX = result['Name']

    PX = np.array(PX)
    TX = np.array(TX).astype(int)
    DX = np.array(DX)

    progX, difX, totX, progerrX, diferrX, toterrX, ttX = reset_means(PX, DX, TX)

    '''plt.figure(1, figsize=(11, 6))
    plt.suptitle('Results for %s' % signal, fontsize=20)
    plt.subplot(221)

    plt.plot(T+np.random.normal(loc=0, scale=.2, size=len(T)),
             P+D, color='#000000', marker='.', markersize=8, linewidth=0, label='Total_Cyclo', alpha=.6)
    plt.plot(T+np.random.normal(loc=0, scale=.2, size=len(T)), D, 'olive',
             marker='.', markersize=8, linewidth=0, label='Dif_Cyclo', alpha=.6)
    plt.plot(T+np.random.normal(loc=0, scale=.2, size=len(T)), P, 'goldenrod',
             marker='.', markersize=8, linewidth=0, label='Prog_Cyclo', alpha=.6)
    plt.errorbar(tt, tot, yerr=toterr, color='#888888', marker='.',
                 markersize=0, linewidth=0, elinewidth=1, capsize=5)
    plt.errorbar(tt, prog, yerr=progerr, color='goldenrod', marker='.',
                 markersize=0, linewidth=0, elinewidth=1, capsize=5)
    plt.errorbar(tt, dif, yerr=diferr, color='olive', marker='.',
                 markersize=0, linewidth=0, elinewidth=1, capsize=5)
    plt.xlabel('hpf', fontsize=8)
    plt.xticks(tt, tt)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylabel('#objects', fontsize=8)
    plt.ylim(bottom=0)
    plt.title('Breakdown of exp. points')
    plt.legend(loc='best', fontsize=8)
    print('First')'''

    progX = np.array(progX)
    difX = np.array(difX)
    totX = np.array(totX)
    progerrX = np.array(progerrX)
    diferrX = np.array(diferrX)
    toterrX = np.array(toterrX)

    #-------------------------------------------------------------------------
    # DATA TUNING
    #------------------------------------------------------------------------

    #P := Todos los datos de progenitoras;   D := Todos los datos de diferenciadas;   T := Todos los tiempos (repetidos)
    #prog := media de progenitoras a cada tiempo; dif := media de diferenciadas a cada tiempo; tot := media de células totales a cada tiempo;  tt := tiempos (sin repetir)

    #len(P) = len(D) = len(T)
    #len(prog) = len(dif) = len(tot) = len(tt)
    for i in range(len(T)):  # Modify the hpf of certain groups
        if T[i] == 29:
            T[i] = 31
        if T[i] == 30:
            T[i] = 29
        if T[i] == 31:
            T[i] = 30
        if T[i] == 41:
            T[i] = 42
    for i in range(len(TX)):  # Modify the hpf of certain groups
        if TX[i] == 31:
            H = np.copy(PX[i])
            PX[i] = DX[i]
            DX[i] = H

    prog, dif, tot, progerr, diferr, toterr, tt = reset_means(P, D, T)

    progX, difX, totX, progerrX, diferrX, toterrX, ttX = reset_means(PX, DX, TX)

    #-------------------------------------------------------------------------
    # END
    #-------------------------------------------------------------------------

    df = pd.read_csv(
        "C://Users//pablo//Desktop//Física//Master UAM//Master thesis//drugs//Ciclo//output//3D_OSCAR_1//Results//homeo.csv", sep=';')
    df = df.dropna()

    plt.figure(3)
    plt.subplot(121)
    plt.plot(T+np.random.normal(loc=0, scale=.2, size=len(T)),
             P+D, color='#000000', marker='.', markersize=8, linewidth=0, label='Total', alpha=.6)
    plt.plot(T+np.random.normal(loc=0, scale=.2, size=len(T)), D, 'olive',
             marker='.', markersize=8, linewidth=0, label='Diffentiated', alpha=.6)
    plt.plot(T+np.random.normal(loc=0, scale=.2, size=len(T)), P, 'goldenrod',
             marker='.', markersize=8, linewidth=0, label='Progenitors', alpha=.6)
    plt.errorbar(tt, tot, yerr=toterr, color='#888888', marker='.',
                 markersize=0, linewidth=0, elinewidth=1, capsize=5)
    plt.errorbar(tt, prog, yerr=progerr, color='goldenrod', marker='.',
                 markersize=0, linewidth=0, elinewidth=1, capsize=5)
    plt.errorbar(tt, dif, yerr=diferr, color='olive', marker='.',
                 markersize=0, linewidth=0, elinewidth=1, capsize=5)
    plt.xlabel('hpf', fontsize=16)
    plt.xticks(tt, tt)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('#objects', fontsize=16)
    plt.ylim(bottom=0)
    plt.legend(loc='best', fontsize=16)
    plt.title('Cyclopamine+', fontsize=18)

    plt.subplot(122)
    plt.plot(TX+np.random.normal(loc=0, scale=.2, size=len(TX)),
             PX+DX, color='#000000', marker='.', markersize=8, linewidth=0, label='Total', alpha=.6)
    plt.plot(TX+np.random.normal(loc=0, scale=.2, size=len(TX)), DX, '#880088',
             marker='.', markersize=8, linewidth=0, label='Differentiated', alpha=.6)
    plt.plot(TX+np.random.normal(loc=0, scale=.2, size=len(TX)), PX, 'orangered',
             marker='.', markersize=8, linewidth=0, label='Progenitors', alpha=.6)
    plt.errorbar(ttX, totX, yerr=toterrX, color='#888888', marker='.',
                 markersize=0, linewidth=0, elinewidth=1, capsize=5)
    plt.errorbar(ttX, progX, yerr=progerrX, color='orangered', marker='.',
                 markersize=0, linewidth=0, elinewidth=1, capsize=5)
    plt.errorbar(ttX, difX, yerr=diferrX, color='#880088', marker='.',
                 markersize=0, linewidth=0, elinewidth=1, capsize=5)
    plt.xlabel('hpf', fontsize=16)
    plt.xticks(ttX, ttX)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(bottom=0)
    plt.legend(loc='best', fontsize=16)
    plt.title('XAV939+', fontsize=18)


    plt.figure(1)
    plt.subplot(243)
    plt.title("Homeostatic")
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylabel('#cells')
    plt.xlabel('hpf')
    plt.errorbar(df.iloc[:, 0], df.iloc[:, 1], yerr=df.iloc[:, 5], color='#000000', marker='s',
                 linewidth=0, elinewidth=1, markersize=4, capsize=5, label='total/Homeo')
    plt.errorbar(df.iloc[:, 0], df.iloc[:, 2], yerr=df.iloc[:, 6], color='#980000',
                 marker='s', elinewidth=1, linewidth=0, markersize=4, capsize=5,  label='prog/Homeo')
    plt.errorbar(df.iloc[:, 0], df.iloc[:, 3], yerr=df.iloc[:, 7], color='#009664',
                 marker='s', elinewidth=1, linewidth=0, markersize=4, capsize=5, label='dif/Homeo')
    plt.ylim(0,17000)
    plt.legend(loc='best', fontsize=8)



    plt.subplot(244)
    plt.title("Homeostatic")
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('hpf')
    plt.errorbar(tt, tot, yerr=toterr, color='#000000', linewidth=0, elinewidth=1,
                 marker='.', markersize=8, capsize=5, label='total/'+signal+'+')
    plt.errorbar(tt, prog, yerr=progerr, color='goldenrod', linewidth=0, elinewidth=1,
                 marker='.', markersize=8, capsize=5, label='prog/'+signal+'+')
    plt.errorbar(tt, dif, yerr=diferr, color='olive', linewidth=0, elinewidth=1,
                 marker='.', markersize=8, capsize=5, label='dif/'+signal+'+')

    plt.legend(loc='best', fontsize=8)
    plt.ylim(0,17000)

    plt.subplot(223)

    x = np.linspace(min(df.iloc[:, 0]), max(df.iloc[:, 0]), 200)

    plt.title('Fitting of Prog. cells', fontsize=12)

    def double_exp(x, a, b, c):
        return a*x**b*np.exp(c*x)
        #return a*np.exp(c*x+b*np.exp(c*x))
    pol, cov1 = curve_fit(double_exp, tt, prog,
                          p0=(200, 0, 0.1), maxfev=5000)
    pol1 = np.copy(pol)
    error_Cyc_prog = delta_prog(pol, cov1, x)

    plt.plot(x, double_exp(x, pol[0], pol[1], pol[2]), 'goldenrod',
             linestyle='--', alpha=1)

    plt.fill_between(x, double_exp(x, pol[0], pol[1], pol[2])+1.96*np.sqrt(error_Cyc_prog), double_exp(
        x, pol[0], pol[1], pol[2])-1.96*np.sqrt(error_Cyc_prog), color='goldenrod', alpha=0.4)
    plt.plot(tt, prog, color='goldenrod', linewidth=0,
                 marker='.', markersize=8, label='P/'+signal+'+')
    plt.ylim(bottom=0)
    fit_Cyc_prog, fit_Cyc_progerr = pol, np.sqrt(np.diag(cov1))


    pol, cov5 = curve_fit(double_exp, ttX, progX,
                          p0=(200, 0, 0.1), maxfev=5000)
    pol5 = np.copy(pol)
    error_XAV_prog = delta_prog(pol5, cov5, x)
    fit_XAV_prog, fit_XAV_progerr = pol, np.sqrt(np.diag(cov5))


    pol, cov2 = curve_fit(
        double_exp, df.iloc[:, 0], df.iloc[:, 2], p0=(200, 0, 0.1), maxfev=5000)

    pol2 = np.copy(pol)
    fit_Hom_prog, fit_Hom_progerr = pol, np.sqrt(np.diag(cov2))
    error_Hom_prog = delta_prog(pol, cov2, x)
    plt.plot(x, double_exp(x, pol[0], pol[1], pol[2]), '#980000',
             linestyle='--', alpha=1)
    plt.fill_between(x, double_exp(x, pol[0], pol[1], pol[2])+1.96*np.sqrt(error_Hom_prog), double_exp(
        x, pol[0], pol[1], pol[2])-1.96*np.sqrt(error_Hom_prog), color='#980000', alpha=0.4)
    plt.errorbar(df.iloc[:, 0], df.iloc[:, 2], yerr=df.iloc[:, 9], color='#980000', linewidth=0, elinewidth=1,
                 marker='s', markersize=4, capsize=5, label='P/Homeo')

    plt.legend(loc='best', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('hpf', fontsize=8)
    plt.ylabel('#objects', fontsize=8)
    print('Third')

    plt.subplot(224)
    plt.title('Fitting of Dif. cells', fontsize=12)

    def simp_exp(x, b, a):
        return a*np.exp(x*b)

    pol, cov4 = curve_fit(
        simp_exp, df.iloc[:, 0], df.iloc[:, 3], p0=(0.2, 3), maxfev=5000)

    pol4 = np.copy(pol)
    error_Hom_dif = delta_dif(pol, cov4, x)

    plt.plot(x, simp_exp(x, pol[0], pol[1]), '#009664',
             linestyle='--')

    plt.fill_between(x, simp_exp(x, pol[0], pol[1])+1.96*np.sqrt(error_Hom_dif), simp_exp(
        x, pol[0], pol[1])-1.96*np.sqrt(error_Hom_dif), color='#009664', alpha=0.4)

    plt.errorbar(df.iloc[:, 0], df.iloc[:, 3], yerr=df.iloc[:, 10], color='#009664', linewidth=0, elinewidth=1,
                 marker='s', markersize=4, capsize=5, label='D/Homeo')
    fit_Hom_dif, fit_Hom_diferr = pol, np.sqrt(np.diag(cov4))

    pol, cov3 = curve_fit(simp_exp, tt, dif, p0=(0.2, 3), maxfev=5000)
    pol3 = np.copy(pol)
    error_Cyc_dif = delta_dif(pol, cov3, x)
    fit_Cyc_dif, fit_Cyc_diferr = pol, np.sqrt(np.diag(cov3))
    plt.plot(x, simp_exp(x, pol[0], pol[1]), 'olive',
             linestyle='--', alpha=1)
    plt.plot(tt, dif, color='olive', linewidth=0,
                 marker='.', markersize=8, label='D/'+signal)
    polA, covA = np.polyfit(tt, np.log(dif), 1, cov=True)

    pol, cov6 = curve_fit(simp_exp, ttX, difX, p0=(0.2, 3), maxfev=5000)
    pol6 = np.copy(pol)
    error_XAV_dif = delta_dif(pol, cov6, x)
    fit_XAV_dif, fit_XAV_diferr = pol, np.sqrt(np.diag(cov6))

    plt.fill_between(x, simp_exp(x, pol[0], pol[1])+1.96*np.sqrt(error_Cyc_dif), simp_exp(
        x, pol[0], pol[1])-1.96*np.sqrt(error_Cyc_dif), color='olive', alpha=0.4)
    '''plt.fill_between(x, np.maximum(simp_exp(x, fit_Hom_dif[0], fit_Hom_dif[1])-1.96*np.sqrt(error_Hom_dif), simp_exp(
        x, fit_Cyc_dif[0], fit_Cyc_dif[1])-1.96*np.sqrt(error_Cyc_dif)), np.minimum(simp_exp(x, fit_Hom_dif[0], fit_Hom_dif[1])+1.96*np.sqrt(error_Hom_dif), simp_exp(
            x, fit_Cyc_dif[0], fit_Cyc_dif[1])+1.96*np.sqrt(error_Cyc_dif)), where=(np.maximum(simp_exp(x, fit_Hom_dif[0], fit_Hom_dif[1])-1.96*np.sqrt(error_Hom_dif), simp_exp(
                x, fit_Cyc_dif[0], fit_Cyc_dif[1])-1.96*np.sqrt(error_Cyc_dif))-np.minimum(simp_exp(x, fit_Hom_dif[0], fit_Hom_dif[1])+1.96*np.sqrt(error_Hom_dif), simp_exp(
                    x, fit_Cyc_dif[0], fit_Cyc_dif[1])+1.96*np.sqrt(error_Cyc_dif))) < 0, color='coral', alpha=0.6, label='Overlap')'''
    plt.ylim(bottom=0)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(loc='best', fontsize=8)
    plt.xlabel('hpf', fontsize=8)
    print('Fourth')

    plt.figure(6)
    plt.subplot(231)

    plt.plot(x, double_exp(x, pol2[0], pol2[1], pol2[2]), '#980000',
             linestyle='--')

    plt.fill_between(x, double_exp(x, pol2[0], pol2[1], pol2[2])+1.96*np.sqrt(error_Hom_prog), double_exp(
        x, pol2[0], pol2[1], pol2[2])-1.96*np.sqrt(error_Hom_prog), color='#980000', alpha=0.4)

    plt.errorbar(df.iloc[:, 0], df.iloc[:, 2], yerr=df.iloc[:, 10], color='#980000', linewidth=0, elinewidth=1,
                 marker='s', markersize=6, capsize=5, label='Control')

    plt.plot(x, double_exp(x, pol1[0], pol1[1], pol1[2]), 'goldenrod',
             linestyle='--', alpha=1)
    plt.errorbar(tt, prog, yerr = progerr, color='goldenrod', linewidth=0, elinewidth = 1, capsize = 5,
                 marker='.', markersize=10, label='Cyclopamine+')
    plt.fill_between(x, double_exp(x, pol1[0], pol1[1], pol1[2])+1.96*np.sqrt(error_Cyc_prog), double_exp(
        x, pol1[0], pol1[1], pol1[2])-1.96*np.sqrt(error_Cyc_prog), color='goldenrod', alpha=0.4)

    plt.ylim(bottom=0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='best', fontsize=16)
    plt.ylabel('# P cells', fontsize=16)
    plt.xticks([20,25,30,35,40,45],[])

    plt.subplot(232)
    plt.plot(x, simp_exp(x, pol4[0], pol4[1]), '#009664',
             linestyle='--')

    plt.fill_between(x, simp_exp(x, pol4[0], pol4[1])+1.96*np.sqrt(error_Hom_dif), simp_exp(
        x, pol4[0], pol4[1])-1.96*np.sqrt(error_Hom_dif), color='#009664', alpha=0.4)

    plt.errorbar(df.iloc[:, 0], df.iloc[:, 3], yerr=df.iloc[:, 10], color='#009664', linewidth=0, elinewidth=1,
                 marker='s', markersize=6, capsize=5, label='Control')

    plt.plot(x, simp_exp(x, pol3[0], pol3[1]), 'olive',
             linestyle='--', alpha=1)
    plt.errorbar(tt, dif, yerr = diferr, color='olive', linewidth=0, elinewidth = 1, capsize = 5,
                 marker='.', markersize=10, label='Cyclopamine+')
    plt.fill_between(x, simp_exp(x, pol3[0], pol3[1])+1.96*np.sqrt(error_Cyc_dif), simp_exp(
        x, pol3[0], pol3[1])-1.96*np.sqrt(error_Cyc_dif), color='olive', alpha=0.4)
    plt.ylim(bottom=0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='best', fontsize=16)
    plt.xticks([20,25,30,35,40,45],[])
    plt.ylabel('# D cells', fontsize=16)

    plt.subplot(234)

    plt.plot(x, double_exp(x, pol2[0], pol2[1], pol2[2]), '#980000',
             linestyle='--')

    plt.fill_between(x, double_exp(x, pol2[0], pol2[1], pol2[2])+1.96*np.sqrt(error_Hom_prog), double_exp(
        x, pol2[0], pol2[1], pol2[2])-1.96*np.sqrt(error_Hom_prog), color='#980000', alpha=0.4)

    plt.errorbar(df.iloc[:, 0], df.iloc[:, 2], yerr=df.iloc[:, 10], color='#980000', linewidth=0, elinewidth=1,
                 marker='s', markersize=6, capsize=5, label='Control')

    plt.plot(x, double_exp(x, pol5[0], pol5[1], pol5[2]), 'orangered',
             linestyle='--', alpha=1)
    plt.errorbar(ttX, progX, yerr = progerrX, color='orangered', linewidth=0, elinewidth = 1, capsize = 5,
                 marker='.', markersize=10, label='XAV939+')
    plt.fill_between(x, double_exp(x, pol5[0], pol5[1], pol5[2])+1.96*np.sqrt(error_XAV_prog), double_exp(
        x, pol5[0], pol5[1], pol5[2])-1.96*np.sqrt(error_XAV_prog), color='orangered', alpha=0.4)

    plt.ylim(bottom=0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='best', fontsize=16)
    plt.ylabel('# P cells', fontsize=16)
    plt.xlabel('hpf',fontsize=16)

    plt.subplot(235)
    plt.plot(x, simp_exp(x, pol4[0], pol4[1]), '#009664',
             linestyle='--')

    plt.fill_between(x, simp_exp(x, pol4[0], pol4[1])+1.96*np.sqrt(error_Hom_dif), simp_exp(
        x, pol4[0], pol4[1])-1.96*np.sqrt(error_Hom_dif), color='#009664', alpha=0.4)

    plt.errorbar(df.iloc[:, 0], df.iloc[:, 3], yerr=df.iloc[:, 10], color='#009664', linewidth=0, elinewidth=1,
                 marker='s', markersize=6, capsize=5, label='Control')

    plt.plot(x, simp_exp(x, pol6[0], pol6[1]), '#880088',
             linestyle='--', alpha=1)
    plt.errorbar(ttX, difX, yerr = diferrX, color='#880088', linewidth=0, elinewidth = 1, capsize = 5,
                 marker='.', markersize=10, label='XAV939+')
    plt.fill_between(x, simp_exp(x, pol6[0], pol6[1])+1.96*np.sqrt(error_XAV_dif), simp_exp(
        x, pol6[0], pol6[1])-1.96*np.sqrt(error_XAV_dif), color='#880088', alpha=0.4)
    plt.ylim(bottom=0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='best', fontsize=16)
    plt.xticks(fontsize=16)
    plt.ylabel('# D cells', fontsize=16)
    plt.xlabel('hpf',fontsize=16)

    #------------------------------------------------------------------
    #BRANCHING PROCESS
    #------------------------------------------------------------------
    num = len(x)
    err1 = delta_deltaprog(fit_Cyc_prog, cov1, x)
    err2 = delta_deltaprog(fit_Hom_prog, cov2, x)
    err5 = delta_deltaprog(fit_XAV_prog, cov5, x)
    err3 = delta_deltadif(fit_Cyc_dif, cov3, x)
    err4 = delta_deltadif(fit_Hom_dif, cov4, x)
    err6 = delta_deltadif(fit_XAV_dif, cov6, x)

    print(fit_Cyc_prog,fit_Cyc_dif,fit_Hom_prog,fit_Hom_dif,fit_XAV_prog,fit_XAV_dif)

    def proge(x, coef):
        return coef[0]*x**coef[1]*np.exp(coef[2]*x)
        #return coef[0]*np.exp(coef[2]*x+coef[1]*np.exp(coef[2]*x))
    def dife(x, coef):
        return coef[1]*np.exp(coef[0]*x)
    A = np.zeros((num,2))

    R1 = np.sqrt(np.sum((P-proge(T,fit_Cyc_prog))**2)/(len(T)-3))
    R3 = np.sqrt(np.sum((D-dife(T,fit_Cyc_dif))**2)/(len(T)-2))
    R2 = np.sqrt(np.sum((df.iloc[:, 2]-proge(df.iloc[:, 0],fit_Hom_prog))**2)/(len(df.iloc[:, 0])-3))
    R4 = np.sqrt(np.sum((df.iloc[:, 3]-dife(df.iloc[:, 0],fit_Hom_dif))**2)/(len(df.iloc[:, 0])-2))
    R5 = np.sqrt(np.sum((PX-proge(TX,fit_XAV_prog))**2)/(len(TX)-3))
    R6 = np.sqrt(np.sum((DX-dife(TX,fit_XAV_dif))**2)/(len(TX)-2))
    print(R1,R2,R3,R4,R5,R6)

    tdif = abs((proge(x,fit_Cyc_prog)-proge(x,fit_Hom_prog))/np.sqrt(error_Cyc_prog+error_Hom_prog))
    df = 9+11
    #df = (error_Cyc_prog/12+error_Hom_prog/14)**2/((error_Cyc_prog/12)**2/10+(error_Hom_prog/14)**2/12)
    ts = stats.t.cdf(tdif,df*np.ones(len(tdif)))

    plt.subplot(233)
    plt.plot(x,2*(1-ts),'goldenrod',label = 'P cells')
    tdif = abs((dife(x,fit_Cyc_dif)-dife(x,fit_Hom_dif))/np.sqrt(error_Cyc_dif+error_Hom_dif))
    df = 10+12
    #df = (error_Cyc_prog/12+error_Hom_prog/14)**2/((error_Cyc_prog/12)**2/10+(error_Hom_prog/14)**2/12)
    ts = stats.t.cdf(tdif,df*np.ones(len(tdif)))

    plt.plot(x,2*(1-ts),'olive',label = 'D cells')
    plt.plot(x, 0.05*np.ones(len(x)), color = 'k')
    plt.yscale('log')
    plt.ylabel('p-value',fontsize=16)
    plt.legend(loc='best', fontsize=16)
    plt.xticks([20,25,30,35,40,45],[])
    plt.yticks(fontsize=16)

    tdif = abs((proge(x,fit_XAV_prog)-proge(x,fit_Hom_prog))/np.sqrt(error_XAV_prog+error_Hom_prog))
    df = 6+11
    #df = (error_Cyc_prog/12+error_Hom_prog/14)**2/((error_Cyc_prog/12)**2/10+(error_Hom_prog/14)**2/12)
    ts = stats.t.cdf(tdif,df*np.ones(len(tdif)))

    plt.subplot(236)
    plt.plot(x,2*(1-ts),'orangered',label = 'P cells')
    tdif = abs((dife(x,fit_XAV_dif)-dife(x,fit_Hom_dif))/np.sqrt(error_XAV_dif+error_Hom_dif))
    df = 7+12
    #df = (error_Cyc_prog/12+error_Hom_prog/14)**2/((error_Cyc_prog/12)**2/10+(error_Hom_prog/14)**2/12)
    ts = stats.t.cdf(tdif,df*np.ones(len(tdif)))

    plt.plot(x,2*(1-ts),'#880088',label = 'D cells')
    plt.plot(x, 0.05*np.ones(len(x)), color = 'k')
    plt.yscale('log')
    plt.xlabel('hpf',fontsize=16)
    plt.ylabel('p-value',fontsize=16)
    plt.legend(loc='best', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    '''
    plt.figure(2)
    plt.subplot(221)
    plt.plot(x, proge(x, fit_Cyc_prog), 'goldenrod')
    plt.errorbar(tt, prog, yerr=progerr, linewidth=0,
                 elinewidth=1, color='goldenrod', marker='.', capsize=5)
    plt.plot(x, dife(x, fit_Cyc_dif), 'olive')
    plt.errorbar(tt, dif, yerr=diferr, linewidth=0,
                 elinewidth=1, color='olive', marker='.', capsize=5)

    x1 = np.linspace(np.min(df.iloc[:, 0]), np.max(df.iloc[:, 0]), 1000)
    plt.plot(x1, proge(x1, fit_Hom_prog), '#980000')
    plt.errorbar(df.iloc[:, 0], df.iloc[:, 2], yerr=df.iloc[:, 9],
                 linewidth=0, elinewidth=1, color='#980000', marker='.', capsize=5)
    plt.plot(x1, dife(x1, fit_Hom_dif), '#009664')
    plt.errorbar(df.iloc[:, 0], df.iloc[:, 3], yerr=df.iloc[:,
                                                            10], linewidth=0, elinewidth=1, color='#009664', marker='.', capsize=5)

    Hom2_prog = (224, -0.01,  0.1)
    Hom2_dif = (0.168, np.log(3.10))
    plt.subplot(222)
    plt.plot(x, proge(x, fit_Cyc_prog), 'goldenrod')
    plt.errorbar(tt, prog, yerr=progerr, linewidth=0,
                 elinewidth=1, color='goldenrod', marker='.')
    plt.plot(x, dife(x, fit_Cyc_dif), 'olive')
    plt.errorbar(tt, dif, yerr=diferr, linewidth=0,
                 elinewidth=1, color='olive', marker='.')

    plt.plot(x1, proge(x1, fit_Hom_prog), '#980000')
    plt.errorbar(df.iloc[:, 0], df.iloc[:, 2], yerr=df.iloc[:, 9],
                 linewidth=0, elinewidth=1, color='#980000', marker='.')
    plt.plot(x1, dife(x1, fit_Hom_dif), '#009664')
    plt.errorbar(df.iloc[:, 0], df.iloc[:, 3], yerr=df.iloc[:,
                                                            10], linewidth=0, elinewidth=1, color='#009664', marker='.')
    '''
    plt.figure(2)
    plt.subplot(211)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    #plt.axvspan(x[1:][mask][0], x[1:][mask][-1], color='#888888', alpha=.2)
    gamma = [0.950570, 0.947139, 0.714254, 0.903924, 0.608896, 0.911342]
    phi = [0, 0, 0, 0.005375, 0.000319, 0.012373]

    gammaa = [(gamma[0]+gamma[1])
             / 2,(gamma[0]+gamma[1])
             / 2,(gamma[0]+gamma[1])
             / 2,(gamma[0]+gamma[1])
             / 2, gamma[2], gamma[3], (gamma[4]+gamma[5])/2]
    phii = [0,0,0,0, phi[2], phi[3], (phi[4]+phi[5])/2]

    gamma_interp = interpolate.splrep([20, 24, 28, 31, 37, 43, 50], gammaa,s=0.005)
    phi_interp = interpolate.splrep([20, 24, 28, 31, 37, 43, 50], phii,s=0.005)
    B = interpolate.splev(x,gamma_interp,der=0)
    A = interpolate.splev(x,phi_interp,der=0)
    for i in range(len(x)):
        if x[i]<31:
            A[i]=0

    final_cyc = Dynamics(proge(x, fit_Cyc_prog), dife(x, fit_Cyc_dif), x, B, A, np.sqrt(err1), np.sqrt(err3), np.sqrt(error_Cyc_prog))
    final_hom = Dynamics(proge(x, fit_Hom_prog), dife(x, fit_Hom_dif), x, np.ones(
        (num))*0.74, np.ones((num))*0, np.sqrt(err2), np.sqrt(err4), np.sqrt(error_Hom_prog))
    final_xav = Dynamics(proge(x, fit_XAV_prog), dife(x, fit_XAV_dif), x, np.ones(
        (num))*0.74, np.ones((num))*0, np.sqrt(err5), np.sqrt(err6), np.sqrt(error_XAV_prog))

    prop_cyc = final_cyc[0]*(np.sqrt(err1)/(proge(x, fit_Cyc_prog)[1:]-proge(x, fit_Cyc_prog)[:-1])+(np.sqrt(err1)+np.sqrt(err3))/(proge(x, fit_Cyc_prog)[1:]-proge(x, fit_Cyc_prog)[:-1]+dife(x, fit_Cyc_dif)[1:]-dife(x, fit_Cyc_dif)[:-1]))
    plt.plot(x[1:], final_cyc[0]+A[1:], 'goldenrod', label='Cyc+')
    plt.plot(x[1:], final_xav[0], 'orangered', label='XAV939+')

    plt.plot(x[1:], final_hom[0],
                   '#009664', label='Control')
    '''plt.plot(x[1:], final_hom[0]+prop_hom,
                   'blue', label='pp-dd/Homeo')
    plt.plot(x[1:], final_hom[0]-prop_hom,
                   'blue', label='pp-dd/Homeo')'''

    #plt.ylim(-1, 1)
    plt.ylabel('pp-dd-$\phi$', fontsize=16)
    plt.legend(loc = 'lower right',fontsize = 16)
    plt.xticks([20,25,30,35,40,45],[])
    plt.yticks(fontsize=16)


    plt.subplot(212)
    plt.plot(x[1:], final_cyc[1], 'olive', label='Cyc+')
    plt.plot(x[1:], final_xav[1], '#880088', label='XAV939+')
    plt.plot(x[1:], final_hom[1], '#980000', label='Control')
    plt.ylabel('T [hrs]', fontsize=16)
    plt.legend(loc = 'upper right',fontsize = 16)
    plt.xlabel('hpf', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    print('MONTECARLO')

    num2 = 10000
    simul_cyc = np.zeros((num2, num-1, 2))
    primer1 = np.zeros((num2, num))
    primer3 = np.zeros((num2, num))
    for j in range(num2):
        permission = False
        while permission == False:
            init1 = np.random.normal(
                proge(x[0], fit_Cyc_prog), np.sqrt(error_Cyc_prog[0]))
            init2 = np.random.normal(
                dife(x[0], fit_Cyc_dif), np.sqrt(error_Cyc_dif[0]))
            primer1[j, 0] = init1
            primer3[j, 0] = init2

            for i in range(num-1):
                rd1 = np.random.normal(
                    proge(x[i+1], fit_Cyc_prog)-proge(x[i], fit_Cyc_prog), np.sqrt(err1[i]))
                rd3 = np.random.normal(
                        dife(x[i+1], fit_Cyc_dif)-dife(x[i], fit_Cyc_dif), np.sqrt(err3[i]))
                if rd1 < -abs(rd3)/2:
                    print(j, x[i], 'Exception')
                    rd1 = -abs(rd3)/2
                primer1[j, i+1] = primer1[j, i] + rd1
                primer3[j, i+1] = primer3[j, i] + rd3

            if len(primer1[j,1:][primer1[j,1:]/primer1[j,:-1]<0])==0:
                permission = True
        simul_cyc[j, :, :] = np.transpose(Dynamics(primer1[j, :], primer3[j, :], x, B, A, np.sqrt(err1), np.sqrt(err3), np.sqrt(error_Cyc_prog))[:2])

        mask = (simul_cyc[j, :, 0] > 1)*(simul_cyc[j, :, 0] < -1)
        if len(mask[mask])>0:
            print([j, 'ohno', np.max(simul_cyc[j, :, 0]),
                  np.min(simul_cyc[j, :, 0])])
            #debug[j, :, :] = np.zeros((num-1, 2), dtype=bool)
        if j % 500 == 0:
            print(j)
    simul_hom = np.zeros((num2, num-1, 2))
    primer2 = np.zeros((num2, num))
    primer4 = np.zeros((num2, num))
    for j in range(num2):
        init1 = np.random.normal(
            proge(x[0], fit_Hom_prog), np.sqrt(error_Hom_prog[0]))
        init2 = np.random.normal(
            dife(x[0], fit_Hom_dif), np.sqrt(error_Hom_dif[0]))
        primer2[j, 0] = init1
        primer4[j, 0] = init2
        for i in range(num-1):
            rd2 = np.random.normal(
                proge(x[i+1], fit_Hom_prog)-proge(x[i], fit_Hom_prog), np.sqrt(err2[i]))
            rd4 = np.random.normal(
                dife(x[i+1], fit_Hom_dif)-dife(x[i], fit_Hom_dif), np.sqrt(err4[i]))
            if rd2 < -abs(rd4)/2:
                rd2 = -abs(rd4)/2
                print(j, x[i], 'Exception')
            primer2[j, i+1] = primer2[j, i] + rd2
            primer4[j, i+1] = primer4[j, i] + rd4
        simul_hom[j, :, :] = np.transpose(Dynamics(primer2[j, :], primer4[j, :], x, np.ones((num))*0.74, np.ones(
            (num))*0, np.sqrt(err2), np.sqrt(err4), np.sqrt(error_Hom_prog))[:2])
        if j % 500 == 0:
            print(j)

    simul_xav = np.zeros((num2, num-1, 2))
    primer5 = np.zeros((num2, num))
    primer6 = np.zeros((num2, num))
    for j in range(num2):
        permission = False
        while permission == False:
            init1 = np.random.normal(
                proge(x[0], fit_XAV_prog), np.sqrt(error_XAV_prog[0]))
            init2 = np.random.normal(
                dife(x[0], fit_XAV_dif), np.sqrt(error_XAV_dif[0]))
            primer5[j, 0] = init1
            primer6[j, 0] = init2

            for i in range(num-1):
                rd5 = np.random.normal(
                    proge(x[i+1], fit_XAV_prog)-proge(x[i], fit_XAV_prog), np.sqrt(err5[i]))
                rd6 = np.random.normal(
                        dife(x[i+1], fit_XAV_dif)-dife(x[i], fit_XAV_dif), np.sqrt(err6[i]))
                if rd5 < -abs(rd6)/2:
                    print(j, x[i], 'Exception')
                    rd5 = -abs(rd6)/2
                if rd5 < 0 and rd5 > rd6:
                    print(j, x[i], 'Exception')
                    rd5 = -abs(rd6)/2
                primer5[j, i+1] = primer5[j, i] + rd5
                primer6[j, i+1] = primer6[j, i] + rd6

            if len(primer5[j,1:][primer5[j,1:]/primer5[j,:-1]<0])==0 and (primer5[j,0]>0 and primer6[j,0]>0):
                permission = True
        simul_xav[j, :, :] = np.transpose(Dynamics(primer5[j, :], primer6[j, :], x, np.ones((num))*0.74, np.ones(
            (num))*0, np.sqrt(err5), np.sqrt(err6), np.sqrt(error_XAV_prog))[:2])
        if j % 500 == 0:
            print(j)

    plt.fill_between(x[1:], np.percentile(simul_cyc[:, :, 1], 84, axis=0),
            np.percentile(simul_cyc[:, :, 1], 16, axis=0), color='olive', alpha=0.5)
    plt.fill_between(x[1:], np.percentile(simul_hom[:, :, 1], 84, axis=0),
            np.percentile(simul_hom[:, :, 1], 16, axis=0), color='#980000', alpha=0.5)
    plt.fill_between(x[1:], np.percentile(simul_xav[:, :, 1], 84, axis=0),
            np.percentile(simul_xav[:, :, 1], 16, axis=0), color='#880088', alpha=0.5)
    plt.ylim(0,18)

    plt.subplot(211)
    plt.fill_between(x[1:], np.percentile(simul_cyc[:, :, 0], 84, axis=0),
                    np.percentile(simul_cyc[:, :, 0], 16, axis=0), color='goldenrod', alpha=0.5)
    plt.fill_between(x[1:], np.percentile(simul_hom[:, :, 0], 84, axis=0),
                    np.percentile(simul_hom[:, :, 0], 16, axis=0), color='#009664', alpha=0.5)
    plt.fill_between(x[1:], np.percentile(simul_xav[:, :, 0], 84, axis=0),
                    np.percentile(simul_xav[:, :, 0], 16, axis=0), color='orangered', alpha=0.5)

    '''plt.subplot(212)
    plt.plot(x,B)
    plt.plot([20,31,37,43,50],gamma,'r.')

    plt.subplot(224)
    plt.plot(x,A)
    plt.plot([20,31,37,43,50],phi,'r.')'''

    plt.figure(4)

    plt.plot(x[1:], simp_exp(x[1:], fit_Cyc_dif[0], fit_Cyc_dif[1]
                             )-simp_exp(x[:-1], fit_Cyc_dif[0], fit_Cyc_dif[1]), 'olive', label='$\Delta D_{Cyc+}$')
    plt.fill_between(x[1:], simp_exp(x[1:], fit_Cyc_dif[0], fit_Cyc_dif[1])-simp_exp(x[:-1], fit_Cyc_dif[0], fit_Cyc_dif[1])+np.sqrt(err3),
                     simp_exp(x[1:], fit_Cyc_dif[0], fit_Cyc_dif[1])-simp_exp(x[:-1], fit_Cyc_dif[0], fit_Cyc_dif[1])-np.sqrt(err3), color='olive', alpha=.4)
    for i in range(5):
        a = int(np.random.rand(1)*num2-1)
        plt.plot(x[1:], primer1[a, 1:]-primer1[a, :-1],
                 color='goldenrod', linewidth=.1, alpha = .5)
        plt.plot(x[1:], primer2[a, 1:]-primer2[a, :-1],
                 color='#980000', linewidth=.1, alpha = .5)
        plt.plot(x[1:], primer3[a, 1:]-primer3[a, :-1],
                 color='olive', linewidth=.1, alpha = .5)
        plt.plot(x[1:], primer4[a, 1:]-primer4[a, :-1],
                 color='#009664', linewidth=.1, alpha = .5)
    plt.plot(x[1:], simp_exp(x[1:], fit_Hom_dif[0], fit_Hom_dif[1]
                             )-simp_exp(x[:-1], fit_Hom_dif[0], fit_Hom_dif[1]), '#009664', label='$\Delta D_{Control}$')
    plt.fill_between(x[1:], simp_exp(x[1:], fit_Hom_dif[0], fit_Hom_dif[1])-simp_exp(x[:-1], fit_Hom_dif[0], fit_Hom_dif[1])+np.sqrt(err4),
                     simp_exp(x[1:], fit_Hom_dif[0], fit_Hom_dif[1])-simp_exp(x[:-1], fit_Hom_dif[0], fit_Hom_dif[1])-np.sqrt(err4), color='#009664', alpha=.4)

    plt.plot(x[1:], double_exp(x[1:], fit_Cyc_prog[0], fit_Cyc_prog[1], fit_Cyc_prog[2])-double_exp(
        x[:-1], fit_Cyc_prog[0], fit_Cyc_prog[1], fit_Cyc_prog[2]), color='goldenrod', label='$\Delta P_{Cyc+}$')
    plt.fill_between(x[1:], double_exp(x[1:], fit_Cyc_prog[0], fit_Cyc_prog[1], fit_Cyc_prog[2])-double_exp(
        x[:-1], fit_Cyc_prog[0], fit_Cyc_prog[1], fit_Cyc_prog[2]) + np.sqrt(err1), double_exp(x[1:], fit_Cyc_prog[0], fit_Cyc_prog[1], fit_Cyc_prog[2])-double_exp(
            x[:-1], fit_Cyc_prog[0], fit_Cyc_prog[1], fit_Cyc_prog[2]) - np.sqrt(err1), color='goldenrod', alpha=0.4)

    plt.plot(x[1:], double_exp(x[1:], fit_Hom_prog[0], fit_Hom_prog[1], fit_Hom_prog[2])-double_exp(
        x[:-1], fit_Hom_prog[0], fit_Hom_prog[1], fit_Hom_prog[2]), color='#980000', label='$\Delta P_{Control}$')
    plt.fill_between(x[1:], double_exp(x[1:], fit_Hom_prog[0], fit_Hom_prog[1], fit_Hom_prog[2])-double_exp(
        x[:-1], fit_Hom_prog[0], fit_Hom_prog[1], fit_Hom_prog[2]) + np.sqrt(err2), double_exp(x[1:], fit_Hom_prog[0], fit_Hom_prog[1], fit_Hom_prog[2])-double_exp(
            x[:-1], fit_Hom_prog[0], fit_Hom_prog[1], fit_Hom_prog[2]) - np.sqrt(err2), color='#980000', alpha=0.4)

    plt.legend(loc='best',fontsize = 18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.xlabel('hpf',fontsize = 18)
    plt.ylabel('# cells',fontsize = 18)

    plt.tight_layout()

    plt.figure(9)
    plt.subplot(121)
    plt.plot(x,B,'turquoise')
    plt.plot([31, 37, 43, 50],gammaa[3:],'k.')
    plt.ylabel('$\gamma$', fontsize = 16)
    plt.xlabel('hpf', fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.subplot(122)
    plt.plot(x,A,'gold')
    plt.plot([31, 37, 43, 50],phii[3:],'k.')
    plt.ylabel('$\phi$',fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.xlabel('hpf', fontsize = 16)
    plt.show()


def reset_means(P, D, T):
    tt = np.unique(T)
    prog = []
    dif = []
    tot = []
    progerr = []
    diferr = []
    toterr = []
    for i in tt:
        num = len(T == i)
        prog.append(np.mean(P[T == i].astype(float)))
        dif.append(np.mean(D[T == i].astype(float)))
        tot.append(np.mean(P[T == i] + D[T == i]))
        progerr.append(np.std(P[T == i].astype(float))/np.sqrt(num))
        diferr.append(np.std(D[T == i].astype(float))/np.sqrt(num))
        toterr.append(np.std(P[T == i] + D[T == i])/np.sqrt(num))

    prog = np.array(prog)
    dif = np.array(dif)
    tot = np.array(tot)
    progerr = np.array(progerr)
    diferr = np.array(diferr)
    toterr = np.array(toterr)
    return prog, dif, tot, progerr, diferr, toterr, tt


def Dynamics(P, D, time, γ, ø, Perr, Derr, perr):
    Pt = P[1:]
    Dt = D[1:]
    t = time[1:]
    q = γ[1:]
    phi = ø[1:]
    Pterr = perr[1:]
    P0 = P[:-1]
    D0 = D[:-1]
    t0 = time[:-1]
    q0 = γ[:-1]
    P0err = perr[:-1]
    #Revisar al añadir quiescencia y muerte
    pp_dd = phi + ((Pt - P0) * (1 - 2 * phi)/(Pt - P0 + Dt - D0))
    T = (t - t0) * np.log(1 + (q*(pp_dd - 1 * phi))) / \
        np.log(Pt / P0)
    for i in range(len(pp_dd)):
        if pp_dd[i] < 0:
            T[i] /= (1 - q[i]*0.8*(pp_dd[i] - phi[i]))
    return [pp_dd, T]

def delta_dif(pol, cov, x):
    X = sp.Symbol('X')
    A = sp.Symbol('A')
    B = sp.Symbol('B')
    simp_exp = A*sp.exp(B*X)
    derA = sp.Derivative(simp_exp, A)
    derB = sp.Derivative(simp_exp, B)
    err = np.zeros(len(x))
    for i in range(len(x)):
        ev = np.zeros((2))
        ev[0] = derB.subs(
            [(X, x[i]), (B, pol[0]), (A, pol[1])])
        ev[1] = derA.subs(
            [(X, x[i]), (B, pol[0]), (A, pol[1])])
        err[i] = \
            (np.matmul(np.matmul(ev, cov), ev.transpose()))
    return err


def delta_prog(pol, cov, x):
    X = sp.Symbol('X')
    A = sp.Symbol('A')
    B = sp.Symbol('B')
    C = sp.Symbol('C')
    doub_exp = A*X**B*sp.exp(C*X)
    #doub_exp = A*sp.exp(C*X+B*sp.exp(C*X))
    derA = sp.Derivative(doub_exp, A)
    derB = sp.Derivative(doub_exp, B)
    derC = sp.Derivative(doub_exp, C)
    err = np.zeros(len(x))
    for i in range(len(x)):
        ev = np.zeros((3))
        ev[0] = derA.subs(
            [(X, x[i]), (A, pol[0]), (B, pol[1]), (C, pol[2])])
        ev[1] = derB.subs(
            [(X, x[i]), (A, pol[0]), (B, pol[1]), (C, pol[2])])
        ev[2] = derC.subs(
            [(X, x[i]), (A, pol[0]), (B, pol[1]), (C, pol[2])])
        err[i] = \
            (np.matmul(np.matmul(ev, cov), ev.transpose()))
    return err


def delta_deltaprog(pol, cov, x):
    A = sp.Symbol('A')
    B = sp.Symbol('B')
    C = sp.Symbol('C')
    X = sp.Symbol('X')
    Y = sp.Symbol('Y')
    #funcion = A*(sp.exp(C*X+B*sp.exp(C*X))-sp.exp(C*Y+B*sp.exp(C*Y)))
    funcion = A*(X**B*sp.exp(C*X)-Y**B*sp.exp(C*Y))
    derA = sp.Derivative(funcion, A)
    derB = sp.Derivative(funcion, B)
    derC = sp.Derivative(funcion, C)
    err = np.zeros(len(x)-1)
    for i in range(len(x)-1):
        ev = np.zeros((3))
        ev[0] = derA.subs(
            [(X, x[i+1]), (A, pol[0]), (B, pol[1]), (C, pol[2]), (Y, x[i])])
        ev[1] = derB.subs(
            [(X, x[i+1]), (A, pol[0]), (B, pol[1]), (C, pol[2]), (Y, x[i])])
        ev[2] = derC.subs(
            [(X, x[i+1]), (A, pol[0]), (B, pol[1]), (C, pol[2]), (Y, x[i])])
        err[i] = \
            (np.matmul(np.matmul(ev, cov), ev.transpose()))
    return err


def delta_deltadif(pol, cov, x):
    A = sp.Symbol('A')
    B = sp.Symbol('B')
    X = sp.Symbol('X')
    Y = sp.Symbol('Y')
    funcion = A*(sp.exp(B*X)-sp.exp(B*Y))
    derA = sp.Derivative(funcion, A)
    derB = sp.Derivative(funcion, B)
    err = np.zeros(len(x)-1)
    for i in range(len(x)-1):
        ev = np.zeros((2))
        ev[0] = derB.subs(
            [(X, x[i+1]), (B, pol[0]), (A, pol[1]), (Y, x[i])])
        ev[1] = derA.subs(
            [(X, x[i+1]), (B, pol[0]), (A, pol[1]), (Y, x[i])])
        err[i] = \
            (np.matmul(np.matmul(ev, cov), ev.transpose()))
    return err


'''
Options for pathways:

'Ciclo' (Hedgehog)
'XAV'   (Wnt)
'Dorso' (BMP)
'''


graph('XAV')
