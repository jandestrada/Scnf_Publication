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

