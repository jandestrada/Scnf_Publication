def get_tptn_data(
                dataframe, 
                real_value='stabilityscore_cnn_calibrated_2classes',           
                predicted_value='stabilityscore_cnn_calibrated_2classes_predictions',
                return_count=True):
    #get all 0's and all 1's
    positives = dataframe[dataframe['stabilityscore_cnn_calibrated_2classes']==1]
    negatives = dataframe[dataframe['stabilityscore_cnn_calibrated_2classes']==0]


    tp = positives[positives['stabilityscore_cnn_calibrated_2classes_predictions']==1]
    tn = negatives[negatives['stabilityscore_cnn_calibrated_2classes_predictions']==0]

    fp = negatives[negatives['stabilityscore_cnn_calibrated_2classes_predictions']==1]
    fn = positives[positives['stabilityscore_cnn_calibrated_2classes_predictions']==0]
    if return_count !=True:
        return tp,tn,fp,fn
    return len(tp),len(tn),len(fp),len(fn)

def get_count_percentiles_list(df,percentiles=[.65,.50,.25,.15]):
    """
    Returns a list of the amount of entropy features in a given percentile 
    Assumes importance values are non-repeating and index of dataframe is 
    sorted by importance. 
    """
    
    spc_features = [
        'S_PC', 'Mean_H_entropy', 'Mean_L_entropy', 'Mean_E_entropy', 
        'Mean_res_entropy', 'SumH_entropies', 'SumL_entropies', 'SumE_entropies', 
        'H_max_entropy', 'H_min_entropy', 'H_range_entropy', 'L_max_entropy', 
        'L_min_entropy', 'L_range_entropy', 'E_max_entropy', 'E_min_entropy', 
        'E_range_entropy']
    count_percentiles_list = []
    unique_importances = df['Importance'].unique()
    for i in range(len(percentiles)):
        
        #this gives the inx of the cutoff value for the percentile
        min_val_inx = round(percentiles[i]*len(unique_importances))
        
        #the cutoff value for the percentile
        min_val = unique_importances[min_val_inx]
        
        #get the subset of rows that make the percentile cutoff
        df_percentile = df[df['Importance']>= min_val]

        #get number of entropy features in subset
        count_percentile = len(
            df_percentile[df_percentile['Feature'].isin(spc_features)])

        #add it to the list
        count_percentiles_list.append(count_percentile)
    
    return count_percentiles_list
