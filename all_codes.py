
setwd("C:/Users/54651/Downloads/consumers project")
# including required libraries in local scope
require(simsAzure)
require(glue)
# loading Azure Credentials -- requires you be logged in via the Azure CLI
Credential <- DefaultAzureCredential(exclude_powershell_credential = TRUE,
                                     exclude_managed_identity_credential = TRUE,
                                     exclude_environment_credential = TRUE,
                                     exclude_interactive_browser_credential = TRUE,
                                     exclude_visual_studio_code_credential = TRUE)

# Query cross-reference data for a batch of meters
xref <- GetXRef(
  AccountName = "derconsumersami",
  Credential = Credential,
  PartitionKey = 1,
  TimeZone = "America/New_York",
  TableName = "AMIXRef"
)
# Writing Xref to file
path_out = 'C:\\Users\\54651\\Downloads\\consumers project'
fileName = paste(path_out, '\\my_file_xref.csv',sep = '')
write.csv(xref, fileName, row.names = FALSE)



xref <- GetXRef(
  AccountName = "derconsumersami",
  Credential = Credential,
  PartitionKey = 1,
  TimeZone = "America/New_York",
  TableName = "AMIXRefNWA"
)
# Writing Xref to file
path_out = 'C:\\Users\\54651\\Downloads\\consumers project'
fileName = paste(path_out, '\\my_file_xref_nwa.csv',sep = '')
write.csv(xref, fileName, row.names = FALSE)

setwd("C:/Users/54651/Downloads/consumers project")

for (i in 2:41){
  data <- GetAMIData(
    AccountName = "derconsumersami",
    Credential = Credential,
    PartitionKey = i,
    RowKey = NULL,
    ColumnName = "AMIData",
    StartDate = lubridate::ymd("2019-01-01"),
    EndDate = lubridate::ymd("2019-12-01"),
    TablePrefix = "RawHourly",
    Uncompress = FALSE,
    Verbose = FALSE
  )
write_json(data,glue("pk_",i,".json"))
}


##############
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob

def daterange(start_date, end_date):
    delta = timedelta(hours=1)
    while start_date < end_date:
        yield start_date
        start_date += delta


def reshape2daily(df_in):
    """
    DESCRIPTION: Reshapes input dataframe with hourly resolution to a dataframe with daily resolution,
                 where every column contains data for a particular hour & different consumers are stacked verticaly.

    Input:
    df_in (pd.DataFrame): Dataframe holding time series data with hourly resolution in every column. Indices are timestamps.

    Output:
    df (pd.DataFrame): Dataframe holding time series data with daily resolution and hours in every columns.

    """
    df = (pd.pivot_table(df_in,
                         values="y",
                         index=[df_in.timestamp.dt.to_period("D"), df_in.id],
                         columns=df_in.timestamp.dt.strftime("%H:%M")
                         )
            .reset_index()
            .sort_values(["id", "timestamp"])
            .set_index("timestamp")
         )
    return df


data_list =  glob.glob("*.json")
summer_loadprofile = []
winter_loadprofile = []
start_date = datetime(2019, 1, 1, 00, 00)
end_date = datetime(2020, 1, 1, 00, 00)

timestamps = []
for single_date in daterange(start_date, end_date):
    timestamps.append(single_date.strftime("%Y-%m-%d %H:%M"))

for file_name in data_list:
    data = pd.read_json(file_name)
    data = data[['RowKey','AMIData']]
    for i in range(0,data.shape[0]):
        d = pd.DataFrame(data.AMIData[i], columns = ['y'])
        d['timestamp'] = timestamps
        d.insert(2, "id", data.RowKey.values[i])
        d = d[['timestamp', 'id', 'y']]
        d['timestamp'] = pd.to_datetime(d['timestamp'])
        d['y'] = d['y'].apply(pd.to_numeric, errors='coerce')
        X_daily = reshape2daily(d)
        X_daily['month'] = X_daily.index.month

        winter = X_daily[X_daily.month == 1]
        del winter['month']
        winter = winter.reset_index(drop= True)
        winter = pd.DataFrame(winter.mean())
        winter_loadprofile.append(winter.T.values.tolist()[0])

        summer = X_daily[X_daily.month == 7]
        del summer['month']
        summer = summer.reset_index(drop= True)
        summer = pd.DataFrame(summer.mean())
        summer_loadprofile.append(summer.T.values.tolist()[0])

summer_data = pd.DataFrame(summer_loadprofile)
winter_data = pd.DataFrame(winter_loadprofile)
summer_data.to_csv("summer_loadcurves.csv")
winter_data.to_csv("winter_loadcurves.csv")

#############

import csv
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
mpl.rcParams['figure.facecolor'] = "w"

def load_data_from_loadshape_file(filename):
    df = pd.read_csv(filename)
    df.index = df.iloc[:,1] # drop unnamed first column
    df = df.dropna() # drop rows with nan
    df = df.iloc[:,2:] # make row_key (premise_id) 
    return df

def check_for_null(df):
    return df.isnull().values.any()
    
file_name = "summer_loadcurves.csv" 
summer = load_data_from_loadshape_file(file_name)
check_for_null(summer)
summarize = summer.T.describe().T
summarize
def min_max_scaling(df):
    df = df.T 
    min_max_scaler = preprocessing.MinMaxScaler()
    return pd.DataFrame(min_max_scaler.fit_transform(df))

scaled_df= min_max_scaling(summer)
scaled_df.describe()
scaled_df = scaled_df.T
scaled_df.index = summer.index
scaled_df


## for plotting
x_axis = pd.date_range("2019-1-1", periods=24, freq="1h").strftime("%H:%M")

weekday_names = ["Mon", "Tue", "Wen", "Thu", "Fri", "Sat", "Sun"]
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul","Aug","Sep","Oct","Nov","Dec"]
quarter_names = ["Q1-weekday","Q2-weekday",  "Q3-weekday", "Q4-weekday","Q1-weekend", "Q2-weekend","Q3-weekend",  "Q4-weekend"]   

def plot_cluster_centroids(df, clust, lw=4, alpha=0.6):
    """
    DESCRIPTION: Plots cluster centroids.
    """
    
    weekly = True if df.shape[1] == 168 else False
    monthly = True if df.shape[1] == 288 else False 
    quarterly = True if df.shape[1] == 192 else False
    if weekly:
        figsize = [15, 4] 
    elif monthly:
        figsize = [25, 4] 
    elif quarterly:
        figsize = [20, 4]
    else:
        figsize = [8, 6]
    
    fontsize = 15
    fig = plt.figure(figsize=figsize)
    
    df.assign(clust=clust).groupby("clust").mean().T.plot(ax=plt.gca(), lw=lw, alpha=alpha);

    plt.title("Cluster Centroids", fontsize=fontsize+5);
    plt.xticks(np.arange(0, len(x_axis), 4), x_axis[::4], fontsize=fontsize);
    
    if weekly:
        plt.xticks(np.arange(0, df.shape[1], 24), 
                   weekday_names, 
                   fontsize=fontsize);
        xposition = np.arange(0, df.shape[1], 24)
        for xc in xposition:
            plt.axvline(x=xc, color='tab:gray', linestyle='--')
    if monthly:
        plt.xticks(np.arange(0, df.shape[1], 24), 
                   month_names, 
                   fontsize=fontsize);
        xposition = np.arange(0, df.shape[1], 24)
        for xc in xposition:
            plt.axvline(x=xc, color='tab:gray', linestyle='--')
    if quarterly:
        plt.xticks(np.arange(0, df.shape[1], 24), 
                   quarter_names, 
                   fontsize=fontsize);
        xposition = np.arange(0, df.shape[1], 24)
        for xc in xposition:
            plt.axvline(x=xc, color='tab:gray', linestyle='--')
    
    plt.yticks(fontsize=fontsize);
    plt.xlabel("")
    plt.ylabel("$P/P_{max}$", fontsize=fontsize)
    plt.legend(title="Cluster centroids:", loc="upper left")
    plt.grid()


def plot_clustered_profiles(df, clust, n_cols=3, alpha=0.2):
    """
    DESCRIPTION: Plots one subplot per cluster, where each subplot contains
                    all profiles in a particular cluster together with a cluster centroid.
    """
    
    weekly = True if df.shape[1] == 168 else False
    monthly = True if df.shape[1] == 288 else False 
    quarterly = True if df.shape[1] == 192 else False

    clust_perc = 100 * clust.value_counts(normalize=True)

    n_rows = np.ceil(clust.nunique() / n_cols)
    
    fontsize = 15

    fig = plt.figure(figsize=[20, n_rows*4])

    for i, clust_n in enumerate(clust_perc.index):

        ax = fig.add_subplot(n_rows, n_cols, i+1)
        df_plot = df[clust == clust_n]
        
        step = 10 if df_plot.shape[0] > 500 else 1  # plot less profiles

        plt.plot(df_plot.iloc[::step].T.values, alpha=alpha, color="dodgerblue")
        df_plot.mean().plot(ax=plt.gca(), alpha=1, color="k", legend=False);

        plt.title("clust: {}, perc: {:.1f}%".format(clust_n, 
                                                    clust_perc.loc[clust_n]), 
                                                    fontsize=fontsize+5);
        plt.xticks(np.arange(0, len(x_axis), 4), x_axis[::4], fontsize=12);

        if weekly:
            plt.xticks(np.arange(0, df.shape[1], 24), 
                       weekday_names, 
                       fontsize=fontsize);
            xposition = np.arange(0, df.shape[1], 24)
            for xc in xposition:
                plt.axvline(x=xc, color='tab:gray', linestyle='--')
        if monthly:
            plt.xticks(np.arange(0, df.shape[1], 24), 
                       month_names, 
                       fontsize=fontsize);
            xposition = np.arange(0, df.shape[1], 24)
            for xc in xposition:
                plt.axvline(x=xc, color='tab:gray', linestyle='--')
        if quarterly:
            plt.xticks(np.arange(0, df.shape[1], 24), 
                       quarter_names, 
                       fontsize=fontsize);
            xposition = np.arange(0, df.shape[1], 24)
            for xc in xposition:
                plt.axvline(x=xc, color='tab:gray', linestyle='--')
        
        plt.yticks(fontsize=fontsize);

        plt.xlabel("Hours", fontsize=fontsize)
        plt.ylabel("$P/P_{max}$", fontsize=fontsize)
        plt.grid()
        plt.savefig('cluster.png')

    plt.tight_layout()


def plot_cost_vs_clusters(df, cluster_algorithm,max_clusters,dtw_metric="dtw"):
    """
    DESCRIPTION: Fits KMeans for different number of clusters & plots cost depending on a number of clusters.
    """
    inertias = []

    for n_clusters in range(2, max_clusters+1):
        if cluster_algorithm == KMeans:
            model = KMeans(n_clusters).fit(df)
        elif cluster_algorithm == TimeSeriesKMeans:
            model = TimeSeriesKMeans(n_clusters,metric = dtw_metric).fit(df) 
        inertias.append(model.inertia_)

    inertias = pd.Series(inertias, index=list(range(2, max_clusters+1)))
    inertias.plot(grid=True);
    plt.xlabel("Number of clusters")
    plt.ylabel("Cost")


## Determine the number of clusters
plt.rcParams['figure.figsize'] = [10, 10]
plot_cost_vs_clusters(scaled_df,KMeans, max_clusters=15)


n_clusters = 8

algorithm = KMeans
model = algorithm(n_clusters).fit(scaled_df)
clust = pd.Series(model.labels_, index=scaled_df.index)
plot_cluster_centroids(scaled_df, clust)


plot_clustered_profiles(scaled_df, clust)

cluster_allocation = pd.DataFrame(clust)
cluster_allocation.columns = ['cluster']
cluster_allocation.index.name = 'row_key'
cluster_allocation


cluster_allocation.to_csv('15032022_clustering_result_summer.csv')

joined = pd.merge(cluster_allocation,summarize, 
                  how = 'inner', 
                  on = "row_key").drop(columns=["count"])
joined

joined.to_csv("15032022_cluster_assignment_to_consumers_dataset.csv")


######################

import pandas as pd
df1 = pd.read_csv("15032022_cluster_assignment_to_consumers_dataset.csv")
df1
df2 = pd.read_csv("AMIXref.csv")
df2.info()
df2 = df2[['AMIDataID','AMIDataID@type','SectorName','SectorName@type']]

df2 = df2.rename(columns={"AMIDataID":'row_key'})
df2
pd.merge(df1, df2, how= 'inner', on = 'row_key').to_csv('cluster_with_address.csv')
df3 = pd.read_csv('my_file_xref_nwa.csv')
df3
df3 = df3[['AMIDataID','SectorName']]
df3 = df3.rename(columns={"AMIDataID":'row_key'})
df3
pd.merge(df1, df3, how= 'inner', on = 'row_key').to_csv('cluster_with_address_from_xrefnwa.csv')

##############


##############################
'''
cluster rule: 
A> super_cluster_1 = cluster_1 + cluster 7
B> super_cluster_2 = cluster_5 + cluster_4
C> super_clsuter_3 = cluster_0
D> super_cluster_4 = cluster_6

'''
import pandas as pd
df1 = pd.read_csv("unique_Partitionkeys_Rowkeys.csv")
df1 = df1[["PartitionKey","RowKey"]]

df2 = pd.read_csv("consumers_cluster_assignment.csv")
df2 = df2[['row_key','cluster']]
df2 = df2.rename(columns = {'row_key':'RowKey'})

df2 = df2[(df2.cluster !=3)&(df2.cluster !=2)]

df2.cluster = df2.cluster.astype('str')
super_cluster_map = {'0':'4',
                     '1':'1',
                    '4':'2',
                    '5':'2',
                    '6':'3',
                    '7':'1'}
    
df2['cluster'] = df2['cluster'].map(super_cluster_map)

df3 = pd.merge(df1,df2, 
               how= 'outer', 
               on= 'RowKey')
df3.to_csv("all_ids_pk_rk_cluster.csv")

###################

df = pd.read_csv("all_ids_pk_rk_cluster.csv",
                usecols =['PartitionKey','RowKey','cluster'])

df= df.dropna(how="any")

cluster_1 =[]
cluster_2 =[]
cluster_3 =[]
cluster_4 =[]
all_data = [cluster_1 ,cluster_2, cluster_3,cluster_4]

for i in df.PartitionKey.unique():
    json_file = "pk_"+str(int(i))+".json" #cosntruct json file name
    db = pd.read_json(json_file) # load associated json file from the directory
    dummy = df[df.PartitionKey==i] # filter all id dataframe for cluster ids in specific PartitionKey
    joined_db = pd.merge(dummy, db, #joined db containes 8760 of each rowkey in specific PartitionKey
                         how = "inner", 
                         on = "RowKey") 
    for clstr in range(1,5): # loading 8760 to specific cluster dataframes
        test = joined_db[joined_db.cluster ==clstr][['AMIData']]
        all_data[clstr-1].append(pd.DataFrame(test.AMIData.tolist()))
        
        
for i in range(len(all_data)):        
    pd.concat(all_data[i]).reset_index(drop=True).to_csv("cluster_{}.csv".format(i))    

####################################


import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
import pylab


def remove_outlier_IQR(df):
    Q1=df.quantile(0.25)
    Q3=df.quantile(0.75)
    IQR=Q3-Q1
    return df[~((df<(Q1-1.5*IQR)) | (df>(Q3+1.5*IQR)))]

files = ['cluster_{}.csv'.format(i) for i in range(0,4)]
mu_values = []

for file in files:
    print(file)
    df = pd.read_csv(file)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    print(df.info())
    
    df = df.interpolate(method='linear', limit_direction='forward', axis=0) #filling missing values
    df = df.fillna(0) # replacing nan-values with zero
    
    print("---before outlier---")
    row_sums = df.sum(axis=1) # getting row sums, since each row is a 8760
    row_sums.to_csv(file[0:9]+"_"+".csv")
    print(shapiro(row_sums.values))
    
    sqrt_row_sums = numpy.sqrt(row_sums)
    #stats.probplot(sqrt_row_sums.values, dist = "norm", plot = pylab)
    #pylab.show()

    print("--after outlier--")
    outlier_removed = remove_outlier_IQR(sqrt_row_sums)
    print(shapiro(outlier_removed.values))
    
    #stats.probplot(outlier_removed.values, dist = "norm", plot = pylab)
    #pylab.show()
    
    print("--fitting a distribution --")
    # Fit a normal distribution to the data:
    mu, std = norm.fit(outlier_removed.values)
    mu_values.append(mu)

    # Plot the histogram.
    plt.hist(outlier_removed.values, bins=25, density=True, alpha=0.6, color='g')
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)
    plt.show()


for file in files:
    print(file)
    df = pd.read_csv(file)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    print(df.info())
    
    df = df.interpolate(method='linear', limit_direction='forward', axis=0) #filling missing values
    df = df.fillna(0) # replacing nan-values with zero
    
    row_sums = df.sum(axis=1) # getting row sums, since each row is a 8760
    row_sums.to_csv(file[0:9]+"_rowsums_"+".csv")

###########################################


import numpy as np
import matplotlib.pyplot as plt

row_sum_files = ["cluster_{}_rowsums_.csv".format(i) for i in range(0,4)]
temp = 0

for file in row_sum_files:
    data_frame = pd.read_csv(file)
    data = (data_frame.iloc[:,1].values)
    count, bins_count = np.histogram(data, bins=100)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    plt.plot(bins_count[1:], cdf, label="CDF")
    plt.legend()
    plt.show()


import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

row_sum_files = ["cluster_{}_rowsums_.csv".format(i) for i in range(0,4)]
temp = 0

for file in row_sum_files:
    data_frame = pd.read_csv(file)
    data = np.sqrt(data_frame.iloc[:,1].values)
    count, bins_count = np.histogram(data, bins=100)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    plt.plot(bins_count[1:], cdf, label="CDF")
    plt.legend()
    plt.show()

#######################################################

mu_values = [81.72901438611422, 89.73593366756255, 76.8293067046773, 84.8230695078433]
cluster_mean_kwh_annual = [i**2 for i in mu_values] 
row_sum_files = ["cluster_{}_rowsums_.csv".format(i) for i in range(0,4)]
temp = 0
print("index,actual, distribution_mean")
for sum_file in row_sum_files:
    data_frame = pd.read_csv(sum_file)
    a = data_frame.iloc[:,1].values
    index_of_mean, value_closest_to_mean =  min(enumerate(a), key=lambda x: abs(x[1]-cluster_mean_kwh_annual[temp]))
    print(index_of_mean,",", value_closest_to_mean,",", cluster_mean_kwh_annual[temp])
    loadshape = pd.read_csv("cluster_{}_noMissingValue_.csv")
    temp = temp+1

#https://stackoverflow.com/questions/65988123/how-to-compute-the-percentiles-from-a-normal-distribution-in-python
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt


mu = [81.73, 89.74, 76.83, 84.82]
sigma = [17.82, 19.88, 19.02, 20.27]

for i in range(4):
    # define the normal distribution and PDF
    dist = sps.norm(loc=mu[i], scale=sigma[i])
    x = np.linspace(dist.ppf(.001), dist.ppf(.999))
    y = dist.pdf(x)

    # calculate PPFs
    ppfs = {}
    for ppf in [.1, .5, .6, .7, .8, .9]:
        p = dist.ppf(ppf)
        ppfs.update({ppf*100: p})

    # plot results
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, color='k')
    for i, ppf in enumerate(ppfs):
        ax.axvline(ppfs[ppf], color=f'C{i}', label=f'{ppf:.0f}th: {ppfs[ppf]:.1f}')
    ax.legend()
    plt.show()

####################

mu_values =  [81.7,  89.7,  76.8,  84.8] #pcntile_50
pcntile_60 = [86.2,  94.8,  81.6,  90  ]
pcntile_80 = [96.7,  106.5, 92.8,  101.9]
pcntile_90 = [104.6, 115.2, 101.2, 110.8]

aggregation_start = [81.7, 89.7,  81.6,  84.8] # taking 50th for cluster #0 and #1, 60th for clust 2, 50th for cluster#3] 
aggregation_end =   [96.7, 106.5, 101.2, 101.9] # taking 80th for clster #0 and #1, 90th for #2, 80th for #3]

lower_kwh_annual = [i**2 for i in aggregation_start]
upper_kwh_annual = [i**2 for i in aggregation_end ]

row_sum_files = ["cluster_{}_rowsums_.csv".format(i) for i in range(0,4)]

holders = []
temp = 0

for file in row_sum_files:
    data_frame = pd.read_csv(file)
    #data_frame= data_frame.iloc[:,1]
    data_frame["delta"] = data_frame.iloc[:,1]-lower_kwh_annual[temp]
    print(lower_kwh_annual[temp])
    #data_frame[data_frame["delta"]<10 & data_frame["delta"]>-10]
    df = data_frame[data_frame["delta"]>lower_kwh_annual[temp]]
    df = df[df["delta"]<=upper_kwh_annual[temp]]
    holders.append(df.index.values)
    temp = temp+1
    
loadshape_repo = ["cluster_{}_noMissingValue_.csv".format(i) for i in range(0,4)]
for i in range(4):
    dummy = []
    df = pd.read_csv(loadshape_repo[i])
    for x in range(0, len(holders[i])): 
        d = df.iloc[holders[i][x]][1:]
        dummy.append(d.values)
    df_1 = pd.DataFrame(dummy)
    df_1.mean().to_csv("final_files/v10/representative_loadshape_for_cluster_{}_v10.csv".format(i))

def plot_8760_loadshapes(folder_name):
    
    loadshape_repo = ["/representative_loadshape_for_cluster_{}_".format(i) for i in range(0,4)]
    loadshape_repo= [str("final_files/"+folder_name+loadshape_repo[i]+folder_name+".csv") for i in range(0,4)]
    #fig, axs = plt.subplots(4,1, figsize=(15, 6), facecolor='w', edgecolor='k')

    dti = pd.date_range(pd.to_datetime(datetime.datetime(2023,1,1)), periods=8760, freq="H", tz = 'US/Central')


    # define subplot grid
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle("loadshape_8760", fontsize=18, y=0.95)

    # loop through tickers and axes
    for file, ax in zip(loadshape_repo, axs.ravel()):
        df = pd.read_csv(file)
        print(file)
        df = df.iloc[:,1]
        df = pd.DataFrame(df.values, columns = [file])
        df.index = dti
        #df.to_csv("final_files/v2/"+file)
        # filter df for ticker and plot on specified axes
        df.plot(ax=ax)

        # chart formatting
        ax.set_title(file)
        ax.get_legend().remove()
        ax.set_xlabel("")
        ax.set_ylim(bottom = 0, top = 6)
        ax.grid()

    plt.show()
    

def plot_summer_loadcurve(folder_name):
    loadshape_repo = ["/representative_loadshape_for_cluster_{}_".format(i) for i in range(0,4)]
    loadshape_repo= [str("final_files/"+folder_name+loadshape_repo[i]+folder_name+".csv") for i in range(0,4)]
    dti = pd.date_range(pd.to_datetime(datetime.datetime(2023,1,1)), periods=8760, freq="H", tz = 'US/Central')
    # define subplot grid
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle("loadshape_8760", fontsize=18, y=0.95)

    # loop through tickers and axes
    for file, ax in zip(loadshape_repo, axs.ravel()):
        df = pd.read_csv(file)
        print(file)
        df = df.iloc[:,1]
        df = pd.DataFrame(df.values, columns = [file])
        df.index = dti
        df['month'] = df.index.month
        df['hour'] = df.index.hour
        df = df[df.month==8]
        dummy = df.groupby(['hour']).mean()
        dummy.iloc[:,0].plot(ax= ax)
        #df.to_csv("final_files/validation/"+file[len(file)-16:len(file)])

        # chart formatting
        ax.set_title(file[29:])
        #ax.get_legend().remove()
        ax.set_xlabel("")
        ax.grid()

    
target = 'v10'
plot_8760_loadshapes(folder_name = target)
plot_summer_loadcurve(folder_name = target)

#################################################################
def summary_of_8760_loadshapes(folder_name):
    
    loadshape_repo = ["/representative_loadshape_for_cluster_{}_".format(i) for i in range(0,4)]
    loadshape_repo= [str("final_files/"+folder_name+loadshape_repo[i]+folder_name+".csv") for i in range(0,4)]

    # loop through tickers and axes
    for file in loadshape_repo:
        df = pd.read_csv(file)
        df = df.iloc[:,1]
        df = pd.DataFrame(df.values, columns = [file])
        print(df.sum())
    
summary_of_8760_loadshapes(folder_name = 'v10')
###############################################################

folder = 'v10'
loadshape_repo = ["/representative_loadshape_for_cluster_{}_".format(i) for i in range(0,4)]
loadshape_repo= [str("final_files/"+folder_name+loadshape_repo[i]+folder_name+".csv") for i in range(0,4)]
dti = pd.date_range(pd.to_datetime(datetime.datetime(2023,1,1)), periods=8760, freq="H", tz = 'US/Central')

all_data = []
k = 0
for i in loadshape_repo:
    df = pd.read_csv(i)
    df = df.iloc[:,1:]
    df.columns = ['site_load']
    df['datetime'] = dti
    df.index = df['datetime']
    df['date'] = pd.DatetimeIndex(df['datetime']).date
    dummy1 = df.groupby("date").sum().reset_index()
    dummy1 = dummy1.rename(columns = {"site_load":"kwh_cluster_"+str(k)})
    dummy2 = df.groupby("date").max().reset_index()
    dummy2 = dummy2.rename(columns = {"site_load":"kw_cluster_"+str(k)})
    dummy = pd.merge(dummy1,dummy2, how = 'left', on = "date" )
    all_data.append(dummy)
    k= k+1

d= pd.concat(all_data, axis=1)
d.to_csv("v10_all_cluster_dailyKWH_dailypeakKW.csv")

df = pd.read_csv("v10_all_cluster_dailyKWH_dailypeakKW.csv")
df = df.iloc[:,1:]
for i in range(0,4):
    kwh_column_name = 'kwh_cluster_{}'.format(i)
    kw_column_name = 'kw_cluster_{}'.format(i)
    column_name_1 = 'cluster_{}_daily_kwh_coverage'.format(i)
    df[column_name_1] = 20/df[kwh_column_name]
    column_name_2 = 'cluster_{}_daily_kw_coverage'.format(i)
    df[column_name_2] = df[kw_column_name]<5
    df[column_name_2] = df[column_name_2].astype(int)

def describe(df, stats):
    d = df.describe()
    return d.append(df.reindex(d.columns, axis = 1).agg(stats))
d = describe(df, ['sum'])
d.T[8:][["mean","sum"]]

'''

mean    sum
cluster_0_daily_kwh_coverage    0.517445    188.867604
cluster_0_daily_kw_coverage 1.000000    365.000000
cluster_1_daily_kwh_coverage    0.432279    157.781656
cluster_1_daily_kw_coverage 1.000000    365.000000
cluster_2_daily_kwh_coverage    0.578483    211.146179
cluster_2_daily_kw_coverage 1.000000    365.000000
cluster_3_daily_kwh_coverage    0.478093    174.503852
cluster_3_daily_kw_coverage 1.000000    365.000000
'''
###################