

def get_plots(audit_results,topology_results=False):
    import matplotlib.pyplot as plt
    import pandas as pd
    from itertools import combinations
    import numpy as np

    plots = {}

    #Distribution of Importances
    fig1,ax1 = plt.subplots(figsize=[20,8])
    plt.hist(audit_results['Importance'])
    ax1.set_title("Distribution of Importance Values",{'fontsize': 20}
    )
    ax1.set_xlabel("Importance",{'fontsize':10})
    ax1.set_ylabel("Number of Features")
    for i in range(10):
        ax1.get_children()[i].set_color("C%i"%i)

    plots['Importances Distribution'] = fig1

    #Horizontal Bar Plot
    fig2,ax2 = plt.subplots(figsize=[20,100])
    ax2.set_title("Ranked Feature Importances",{'fontsize': 20})
    plt.barh(y=audit_results['Importance'].index,width=audit_results['Importance'],
             tick_label=audit_results['Feature'],color='cadetblue')
    ax2.set_xlabel("Importance")

    plt.yticks(fontsize=24)


    plots['Horizontal Bar Plot'] = fig2
        
    if topology_results!= False:
        #Topology Specific Scatter Plot (probably not cause too specific to data)
            
        #Global Highest Importance per Topology (too specific too)
        plots.update(get_global_highest_importance_per_topology(audit_results))
        
    return plots

def get_global_highest_importance_per_topology(audit_results,color='teal'):
    import matplotlib.pyplot as plt
    import pandas as pd
    from itertools import combinations
    import numpy as np
    
    #a dictionary containing the count of each topology
    topcount_dict = audit_results['Topology'].value_counts().to_dict()
    #make the dictionary into a dataframe
    topcount_data = pd.DataFrame(topcount_dict,index=[0])
    
    fig, ax = plt.subplots(figsize=[8,6])
    ax.set_title("Global Highest Importance Feature Count per Topology",{"fontsize":16})
    ax.set_ylabel("Highest Feature Importance Count")
    plt.bar(x=topcount_data.columns,height=topcount_data.iloc[0,:],color=color)
    #fig.savefig("/home/jupyter/tacc-work/model_building/dataframes/saved_plots/shap/topology_specific_plots_rocklin_features/GlobalHighestImportance_oldfeatures.jpeg")
    plt.show()
    
    return {'Global-Highest-Importance':fig}

def topology_specific_scatter_plots(list_of_topology_names,list_of_dataframes,output_path=None):
    import matplotlib.pyplot as plt
    import pandas as pd
    from itertools import combinations
    import numpy as np
    lst = list_of_topology_names

    topologies_list= fix_names_list(list_of_topology_names)

    dataframes_acc = list_of_dataframes
    zipped_dataframes_acc = zip_dataframes(topologies_list,dataframes_acc)


    for df_pair in combinations(zipped_dataframes_acc,2):
        topology_x = df_pair[0][1]
        topology_y = df_pair[1][1]
        x_sorted = df_pair[0][0].sort_values("Feature")
        y_sorted = df_pair[1][0].sort_values("Feature")
        x = x_sorted['Importance']
        y = y_sorted['Importance']

        features_list = x_sorted['Feature']


        #plt.figure(figsize=[13,10])
        fig,ax = plt.subplots(figsize=[13,10])
        name = "%s vs %s Feature Importances"%(topology_x,topology_y)
        plt.title(name,{'fontsize':24})

        ax.scatter(x,y)
        plt.xlabel("%s Importances"%topology_x)
        plt.ylabel("%s Importances"%topology_y)

        #add trendline for y values
        lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        # now plot both limits against eachother
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        #label the most important feature in each
#         x_max = max(x)
#         x_max_index = x[x == x_max].index[0]

#         y_max = max(y)
#         y_max_index = y[y == y_max].index[0]

#         plt.text(x[x_max_index],y[x_max_index],s=features_list[x_max_index]+'_X Max') #label the most important X feature
#         plt.text(x[y_max_index],y[y_max_index],s=features_list[y_max_index]+'_Y Max') #label the most important y feature

        plt.legend()
        if output_path != None:
            fig.savefig(output_path+name.replace(" ","_")+'.jpeg')

def topology_specific_feature_importance(zipped_dataframe,v=False):    
    import matplotlib.pyplot as plt
    import pandas as pd
    from itertools import combinations
    import numpy as np    
    
    #get a list of most important features per topology
    feature_importance_data = []

    for i in range(len(zipped_dataframe)):

        df_tuple_i = zipped_dataframe[i]

        #create a list of the dataframes that does not include the current instance
        temp_list = zipped_dataframe.copy()
        temp_list.pop(i)


        #first dataframe that will be compared
        df_i = df_tuple_i[0].sort_values("Feature")
        df_i_name = df_tuple_i[1]
        #print("Currenlty in %s \n"%df_i_name)

        #list of more important features across all topologies
        most_important_features = df_i['Feature'].tolist()
        if v:
            print("Initially, %i most important features"%(len(most_important_features)))
        for df_tuple_j in temp_list:
            
            
            
            #the dataframe it will be compared to
            df_j = df_tuple_j[0].sort_values("Feature")
            df_j_name = df_tuple_j[1]
            if v:
                print("\tI am comparing %s to %s"%(df_i_name,df_j_name) )
#             print("df_i index: %s"%df_i['Feature'][:5])
#             print("df_j index: %s"%df_j['Feature'][:5])
            
            #all features that were more important for df_i
            current_features = df_i[df_i['Importance'] - df_j["Importance"] >0]['Feature'].tolist()
            if v:
                print("CURRENT FEATURES:(%i)\n%s"%(len(current_features),current_features))
            
            
            #keep only the values that intersect
            most_important_features = list(set(most_important_features).intersection(current_features))
            if v:
                print("MOST IMPORTANT SO FAR:(%i)\n%s"%(len(most_important_features),most_important_features) )
        if v:
            print("Finally, %i most important features"%(len(most_important_features)))
        feature_importance_data.append((df_i_name,most_important_features))
    return feature_importance_data

def get_most_important_features_per_topology(zipped_df,drop_duplicates=True):
    import matplotlib.pyplot as plt
    import pandas as pd
    from itertools import combinations
    import numpy as np 
    df = pd.DataFrame()
    for df_tuple in zipped_df:
        df_name = df_tuple[1]
        df_topology = df_tuple[0]

        df_topology['Topology'] = [df_name]*len(df_topology)

        df = df.append(df_topology,ignore_index=True)
        
        df = df.sort_values('Importance',ascending=False)
        
        if drop_duplicates==True:
            df = df.drop_duplicates(subset='Feature')

    return df


def fix_names_list(list_of_topology_names):
    import matplotlib.pyplot as plt
    import pandas as pd
    from itertools import combinations
    import numpy as np
    
    
    fixed_list= []
    for i in list_of_topology_names:
        if len(i)<1:
            continue
        fixed_list.append(i.split("_acc")[0])
    return fixed_list

def zip_dataframes(list_of_topology_names,list_of_dataframes):
    import matplotlib.pyplot as plt
    import pandas as pd
    from itertools import combinations
    import numpy as np
    
    lst = list_of_topology_names
    if "_acc" in lst:
        lst = fix_names_list(lst)
    
    zipped_dataframes = list(zip(list_of_dataframes, lst))
    return zipped_dataframes
    