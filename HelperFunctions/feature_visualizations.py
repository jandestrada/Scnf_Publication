def show_all_plots(
                    data,
                    features,
                    target='stabilityscore_cnn_calibrated_2classes',
                    title_name="Distribution of entropy features across all protein topologies"):
  
 
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    y = data[target] #target
    x = data[features] #features
    plt.close('all')
    
    #show class imbalance
    ax = sns.countplot(y,label="Count")
    F, T = y.value_counts()
    print('Number of Unstable: %i'%F)
    print("number of Stable: %i"%T)
    plt.show()
    plt.close('all')
    
    #show violin plot
    data_dia = y
    data = x
    data_n_2 = (data - data.mean()) / (data.std())              # standardization
    data = pd.concat([y,data_n_2.iloc[:,:18]],axis=1)
    data = pd.melt(data,id_vars='stabilityscore_cnn_calibrated_2classes',
                        var_name='features',
                        value_name=None)
    plt.figure(figsize=(10,10))
    sns.violinplot(x='features', y=None, hue='stabilityscore_cnn_calibrated_2classes',data=data,split=True, inner="quart")
    plt.xticks(rotation=90)
    plt.title(title_name,fontsize='18.5')
    plt.show()
    plt.close('all')
    
    
    #box plot
    plt.figure(figsize=(10,10))
    sns.boxplot(x="features", y=None, hue='stabilityscore_cnn_calibrated_2classes', data=data)
    plt.xticks(rotation=90)
    #plt.legend(title='stabilityscore_cnn_calibrated_2classes',labels=['unstable','stable'])
    plt.title(title_name,fontsize='18.5')
    plt.show()
    plt.close('all')
    
    
    
def topology_specified_plot(
    tuple_of_list_and_metric):
    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    topology_list_df = tuple_of_list_and_metric[0]
    score_metric = tuple_of_list_and_metric[1]
    
    clf_scores_names = [
                'Accuracy','Balanced Accuracy','AUC Score',
                'Average Precision','F1 Score','Precision',
                'Recall']
   
    rgr_scores_names = ['R-Squared','RMSE']
    
    
    #TODO: Fix it so that it only needs one input. Figure out if it is a clf or rgr from the dataframes
    
    metric_list = None
    if score_metric=='clf':
        metric_list = clf_scores_names
    elif score_metric=='rgr':
        metric_list = rgr_scores_names
    else:
        print("Invalid score metric. Choose either 'clf' or 'rgr'")
        return
        
    
    
    
    top_rs = topology_list_df[topology_list_df['model']=='RS']
    top_r = topology_list_df[topology_list_df['model']=='R']
    top_s = topology_list_df[topology_list_df['model']=='S']



    # sort dataframes for plots
    top_rs = top_rs.sort_values(by='Samples In Test',ascending=False)
    top_r = top_r.sort_values(by='Samples In Test',ascending=False)
    top_s = top_s.sort_values(by='Samples In Test',ascending=False)

    for i in range(len(metric_list)):
            sns.set(rc={'figure.figsize':(35,15)})
            sns.set_style('white')

            fig,ax = plt.subplots()

            score_rs = top_rs[metric_list[i]].values
            score_r = top_r[metric_list[i]].values
            score_s = top_s[metric_list[i]].values

            sns.set(rc={'figure.figsize':(35,15)})


            sns.barplot(
                x=top_rs['topology'].values,
                y=top_rs['Samples In Test'],
                ax=ax,
                color='white',
                edgecolor='black',
                linewidth=4)
            sns.barplot(
                x=top_r['topology'].values,
                y=top_r['Samples In Test'],
                ax=ax,
                color='white',
                edgecolor='black',
                linewidth=4)
            sns.barplot(
                x=top_s['topology'].values,
                y=top_s['Samples In Test'],
                ax=ax,color='white',
                edgecolor='black',
                linewidth=4)
            ax2 = ax.twinx()


            sns.pointplot(
                x=top_rs['topology'].values,
                y=score_rs,
                color='blue',
                join=False,
                scale=1)
            sns.pointplot(
                x=top_r['topology'].values,
                y=score_r,
                color='orange',
                join=False,
                scale=1)
            sns.pointplot(
                x=top_s['topology'].values,
                y=score_s,
                color='green',
                join=False,
                scale=1)


            ax2.legend(
                handles=ax2.collections, 
                labels=["R+S","R","S"],
                prop={'size':'20'})


            plt.title("%s per topology"%metric_list[i],fontsize=30)
            ax.set_xlabel("Topology ",fontsize=22)
            ax.set_xticklabels(ax2.get_xticklabels(),fontsize=20)
            plt.ylabel("%s"%metric_list[i],fontsize=22)
            ax.set_ylabel("Number of samples in test group",fontsize=22)
            plt.show()
            #plt.close('all')
            
            
def loo_plot(tuple_with_list_and_metric):

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    dataframe_list = tuple_with_list_and_metric[0]
    metric = tuple_with_list_and_metric[1]


    #choose a metric
    metric_scores = None
    if metric=='clf':
        metric_scores = [
        'Accuracy','Balanced Accuracy','AUC Score',
        'Average Precision','F1 Score','Precision','Recall']
    elif metric=='rgr':
        metric_scores = ['R-Squared','RMSE']
    

    # sort dataframes for plots
    clf_rs = dataframe_list[0].sort_values(by='Samples In Test',ascending=False)
    clf_r = dataframe_list[1].sort_values(by='Samples In Test',ascending=False)
    clf_s = dataframe_list[2].sort_values(by='Samples In Test',ascending=False)

    for i in range(len(metric_scores)):
            sns.set(rc={'figure.figsize':(35,15)})
            sns.set_style('white')
            
            fig,ax = plt.subplots()

            score_rs = clf_rs[metric_scores[i]].values
            score_r = clf_r[metric_scores[i]].values
            score_s = clf_s[metric_scores[i]].values


            sns.barplot(
                x=clf_rs['Test Group'].values,
                y=clf_rs['Samples In Test'],
                ax=ax,color='.75')
            
            sns.barplot(
                x=clf_r['Test Group'].values,
                y=clf_r['Samples In Test'],
                ax=ax,color='.75')
            
            sns.barplot(
                x=clf_s['Test Group'].values,
                y=clf_s['Samples In Test'],
                ax=ax,color='.75')
            ax2 = ax.twinx()


            sns.pointplot(
                x=clf_rs['Test Group'].values,
                y=score_rs,
                color='blue',
                join=False,
                scale=2)
            sns.pointplot(
                x=clf_r['Test Group'].values,
                y=score_r,
                color='orange',
                join=False,
                scale=2)
            sns.pointplot(
                x=clf_s['Test Group'].values,
                y=score_s,
                color='green',
                join=False,
                scale=2)


            ax2.legend(
                handles=ax2.collections, 
                labels=["R+S","R","S"],
                prop={'size':'20'})


            plt.title("%s per left out group"%metric_scores[i],fontsize=30)
            ax.set_xlabel("Topology Left Out",fontsize=22)
            ax.set_xticklabels(ax2.get_xticklabels(),rotation=90,fontsize=20)
            plt.ylabel("%s"%metric_scores[i],fontsize=22)
            ax.set_ylabel("Samples in Test Group",fontsize=22)
            plt.show()
            #plt.close('all')
            
            
#################### HELPER TO HELPER FUNCTIONS #######################
def get_count_percentiles_list(df):
    total_n_features = len(df)
    percentiles = [.65,.5,.25,.15]
    spc_features = [
        'S_PC', 'Mean_H_entropy', 'Mean_L_entropy', 'Mean_E_entropy', 
        'Mean_res_entropy', 'SumH_entropies', 'SumL_entropies', 'SumE_entropies', 
        'H_max_entropy', 'H_min_entropy', 'H_range_entropy', 'L_max_entropy', 
        'L_min_entropy', 'L_range_entropy', 'E_max_entropy', 'E_min_entropy', 
        'E_range_entropy']
    count_percentiles_list = []

    for i in range(len(percentiles)):
        #get the number of features in a given percentile
        n_percentile = round(total_n_features*percentiles[i])

        #print(n_percentile)
        #get subset of dataframe in that percentile
        df_percentile  = df.iloc[:n_percentile,:]

        #get number of entropy features in subset
        count_percentile = len(
            df_percentile[df_percentile['Feature'].isin(spc_features)])

        #add it to the list
        count_percentiles_list.append(count_percentile)
    
    return count_percentiles_list

def check_only_one_true(lst): 
    #stolen gracefully from https://stackoverflow.com/questions/16801322/how-can-i-check-that-a-list-has-one-and-only-one-truthy-value
    true_found = False
    for v in lst:
        if v and not true_found:
            true_found=True
        elif v and true_found:
             return False #"Too Many Trues"
    return true_found

def overall_results(file_path, loo_run,topology_run,general_run, model_description_column,metric):
    import pandas as pd
    
    #check only one is true
    assert check_only_one_true([loo_run,topology_run,general_run]),"Only one of 'loo_run','topology_run', or 'general_run', can be True"
    
    
    # load the leaderboard dataframe
    leaderboard_dataframe = pd.read_html(file_path)[0]

    # make an rs, r, and s subset
    dataframe_rs = leaderboard_dataframe[
        leaderboard_dataframe[model_description_column]=='RS']
    dataframe_r = leaderboard_dataframe[
        leaderboard_dataframe[model_description_column]=='R']
    dataframe_s = leaderboard_dataframe[
        leaderboard_dataframe[model_description_column]=='S']


    #define variables for visualizations
    if loo_run:
        variables_name = 'Test Group'
        plotting_function = loo_plot
        plotting_input = ([dataframe_rs,dataframe_r,dataframe_s],metric)
        
        #print number of datapoints
        print("Printing number of instances per group:")
        print(dataframe_rs[variables_name].value_counts())
        print("")

    elif topology_run:
        variables_name = 'topology'
        plotting_function = topology_specified_plot
        df_topology = make_topologies_list(file_path)
        plotting_input = (df_topology,metric) #TODO: make this a function that will return a topology_df_list 

    elif general_run:
        pass



    #plot function
    print("Plotting Trends:")
    plotting_function(plotting_input)
    print("")
    
    

    # Output a dataframe with the variation in each model
    
    
    
    
def make_topologies_list(file_path_of_results):
    import pandas as pd
    
    
    #make two columns for sorting purposes
    leaderboard = pd.read_html(file_path_of_results)[0]
    leaderboard['topology'] = [i.split()[1] for i in leaderboard['Data and Split Description']]
    leaderboard['model'] = [i.split()[0] for i in leaderboard['Data and Split Description']]
    
    #show how many datapoints we have per topology
    print(leaderboard['topology'].value_counts())
    
    #make a list of available topologies
    topologies_list = leaderboard['topology'].value_counts().index
    
    df_list = []
    for topology in topologies_list:
        subset_df = leaderboard[leaderboard['topology']==topology]
        df_list.append(subset_df)
    final_df = pd.concat(df_list)
    return final_df
    
def rank_features(audit_df,just_entropy=False):
    df = audit_df.copy()
    spc_features = [
        'S_PC', 'Mean_H_entropy', 'Mean_L_entropy', 'Mean_E_entropy', 
        'Mean_res_entropy', 'SumH_entropies', 'SumL_entropies', 'SumE_entropies', 
        'H_max_entropy', 'H_min_entropy', 'H_range_entropy', 'L_max_entropy', 
        'L_min_entropy', 'L_range_entropy', 'E_max_entropy', 'E_min_entropy', 
        'E_range_entropy']
    if just_entropy:
        return df[df['Feature'].isin(spc_features)]
    df['Importance'] = df['Importance'].round(3)
    return df
    
    

