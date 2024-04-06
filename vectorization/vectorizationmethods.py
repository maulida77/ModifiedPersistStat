import numpy as np
#from gudhi import representations, CubicalComplex
#from sklearn.cluster import KMeans
from vectorization.bar_cleaner import bar_cleaner
from vectorization import GetNewMethods 
#import teaspoon.ML.feature_functions as Ff
#from copy import deepcopy
import scipy
import statistics



__all__ = ["GetPersStats",  "GetPersStats_modiv"  
           ]


def GetPersStats(barcode,*p):
    barcode = bar_cleaner(barcode)
    if (np.size(barcode) > 0):
        # Average of Birth and Death of the barcode
        bc_av0, bc_av1 = np.mean(barcode, axis=0)
        # STDev of Birth and Death of the barcode
        bc_std0, bc_std1 = np.std(barcode, axis=0)
        # Median of Birth and Death of the barcode
        bc_med0, bc_med1 = np.median(barcode, axis=0)
        # Intercuartil range of births and death
        bc_iqr0, bc_iqr1 = np.subtract(*np.percentile(barcode, [75, 25], axis=0)) 
        # Range of births and deaths
        bc_r0, bc_r1=np.max(barcode, axis=0) - np.min(barcode, axis=0)
        # Percentiles of births and deaths
        bc_p10_0,bc_p10_1 = np.percentile(barcode, 10, axis=0)
        bc_p25_0,bc_p25_1 = np.percentile(barcode, 25, axis=0)
        bc_p75_0,bc_p75_1 = np.percentile(barcode, 75, axis=0)
        bc_p90_0,bc_p90_1 = np.percentile(barcode, 90, axis=0)

        
        
        avg_barcodes = (barcode[:,1] + barcode[:,0])/2
        # Average of midpoints of the barcode
        bc_av_av = np.mean(avg_barcodes)
        # STDev of midpoints of the barcode
        bc_std_av = np.std(avg_barcodes)
        # Median of midpoints of the barcode
        bc_med_av = np.median(avg_barcodes)
        # Intercuartil range of midpoints
        bc_iqr_av = np.subtract(*np.percentile(avg_barcodes, [75, 25])) 
        # Range of midpoints
        bc_r_av = np.max(avg_barcodes) - np.min(avg_barcodes)
        # Percentiles of midpoints
        bc_p10_av = np.percentile(barcode, 10)  
        bc_p25_av = np.percentile(barcode,25)   
        bc_p75_av = np.percentile(barcode, 75)  
        bc_p90_av = np.percentile(barcode, 90)   


        
        diff_barcode = np.subtract([i[1] for i in barcode], [
                                   i[0] for i in barcode])
        diff_barcode = np.absolute(diff_barcode)
        # Average of the length of Bars
        bc_lengthAverage = np.mean(diff_barcode)
        # STD of length of Bars
        bc_lengthSTD = np.std(diff_barcode)
        # Median of length of Bars
        bc_lengthMedian = np.median(diff_barcode)
        # Intercuartil range of length of the bars
        bc_lengthIQR= np.subtract(*np.percentile(diff_barcode, [75, 25]))
        # Range of length of the bars
        bc_lengthR=np.max(diff_barcode) - np.min(diff_barcode)
        # Percentiles of lengths of the bars
        bc_lengthp10=np.percentile(diff_barcode, 10)
        bc_lengthp25=np.percentile(diff_barcode, 25)
        bc_lengthp75=np.percentile(diff_barcode, 75)
        bc_lengthp90=np.percentile(diff_barcode, 90)



        
        # Number of Bars
        bc_count = len(diff_barcode)
        # Persitent Entropy
        ent = GetNewMethods.Entropy()
        bc_ent = ent.fit_transform([barcode])
        
        bar_stats = np.array([bc_av0, bc_av1, bc_std0, bc_std1, bc_med0, bc_med1,
                              bc_iqr0, bc_iqr1, bc_r0, bc_r1, bc_p10_0, bc_p10_1, 
                              bc_p25_0, bc_p25_1, bc_p75_0, bc_p75_1, bc_p90_0, 
                              bc_p90_1, 
                              bc_av_av, bc_std_av, bc_med_av, bc_iqr_av, bc_r_av, bc_p10_av, 
                              bc_p25_av, bc_p75_av, bc_p90_av, 
                              bc_lengthAverage, bc_lengthSTD, 
                              bc_lengthMedian, bc_lengthIQR, bc_lengthR, bc_lengthp10,  
                              bc_lengthp25,  bc_lengthp75,  bc_lengthp90,                              
                              bc_count, bc_ent[0][0]])

    else:
        bar_stats = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0,0, 0])

    bar_stats[~np.isfinite(bar_stats)] = 0

    return bar_stats



def GetPersStats_modiv(barcode,*p):
    barcode = bar_cleaner(barcode)
    if (np.size(barcode) > 0):
        # Average of Birth and Death of the barcode
        bc_av0, bc_av1 = np.mean(barcode, axis=0)
        # STDev of Birth and Death of the barcode
        # bc_std0, bc_std1 = np.std(barcode, axis=0)
        bc_std0 = statistics.stdev(barcode[:,0])
        bc_std1 = statistics.stdev(barcode[:,1])
        # Intercuartil range of births and death
        bc_iqr0, bc_iqr1 = np.subtract(*np.percentile(barcode, [75, 25], axis=0)) 
        # Range of births and deaths
        bc_r0, bc_r1=np.max(barcode, axis=0) - np.min(barcode, axis=0)
        ##++ Coefficient of Skewness and Kurtosis of Birth and Death of the barcode
        skew0, skew1 = scipy.stats.skew(barcode, axis=0, bias=False)
        kur0, kur1 = scipy.stats.kurtosis(barcode, axis=0, fisher=True, bias=False)
        
        
        avg_barcodes = (barcode[:,1] + barcode[:,0])/2
        # Average of midpoints of the barcode
        bc_av_av = np.mean(avg_barcodes)
        # STDev of midpoints of the barcode
        # bc_std_av = np.std(avg_barcodes)
        bc_std_av = statistics.stdev(avg_barcodes)
        # Intercuartil range of midpoints
        bc_iqr_av = np.subtract(*np.percentile(avg_barcodes, [75, 25])) 
        # Range of midpoints
        bc_r_av = np.max(avg_barcodes) - np.min(avg_barcodes)
        ##++ Coefficient of Skewness and Kurtosis of Birth and Death of the barcode
        skew_av = scipy.stats.skew(avg_barcodes, bias=False)
        kur_av = scipy.stats.kurtosis(avg_barcodes, fisher=True, bias=False)

        
        diff_barcode = np.subtract([i[1] for i in barcode], [
                                   i[0] for i in barcode])
        diff_barcode = np.absolute(diff_barcode)
        # Average of the length of Bars
        bc_lengthAverage = np.mean(diff_barcode)
        # STD of length of Bars
        # bc_lengthSTD = np.std(diff_barcode)
        bc_lengthSTD = statistics.stdev(diff_barcode)
        # Intercuartil range of length of the bars
        bc_lengthIQR= np.subtract(*np.percentile(diff_barcode, [75, 25]))
        # Range of length of the bars
        bc_lengthR=np.max(diff_barcode) - np.min(diff_barcode)
        ##++ Coefficient of Skewness and Kurtosis of Birth and Death of the barcode
        skew_length = scipy.stats.skew(diff_barcode, bias=False)
        kur_length = scipy.stats.kurtosis(diff_barcode, fisher=True, bias=False)


        # Number of Bars
        bc_count = len(diff_barcode)
        # Persitent Entropy
        ent = GetNewMethods.Entropy()
        bc_ent = ent.fit_transform([barcode])
        
        bar_stats = np.array([bc_av0, bc_av1, bc_std0, bc_std1, 
                              bc_iqr0, bc_iqr1, bc_r0, bc_r1,  skew0, skew1, kur0, kur1,
                              bc_av_av, bc_std_av, bc_iqr_av, bc_r_av, skew_av,  kur_av, 
                              bc_lengthAverage, bc_lengthSTD, 
                              bc_lengthIQR, bc_lengthR, 
                              skew_length, kur_length, 
                              bc_count, bc_ent[0][0]])

    else:
        bar_stats = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                              0, 0, 0,0, 0, 0, 0, 0])

    bar_stats[~np.isfinite(bar_stats)] = 0

    return bar_stats

