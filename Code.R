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



data <- read.csv('C:/Users/54651/Downloads/consumers project/unique_Partitionkeys_Rowkeys.csv')
setwd("C:/Users/54651/Downloads/consumers project/loadshapes")
for (i in 1:nrow(data)){
  AMI_data <- GetAMIData( AccountName = "derconsumersami",
                          Credential = Credential,
                          PartitionKey = data[i,2],
                          RowKey = data[i,3],
                          ColumnName = "AMIData",
                          StartDate = lubridate::ymd("2019-01-01"),  # includes whole month's usage
                          EndDate = lubridate::ymd("2019-12-01"),  # includes whole month's usage
                          TablePrefix = "RawHourly",
                          Uncompress = TRUE,
                          Verbose = FALSE
  )
  write.csv(AMI_data, glue("rowkey_", data[i,3], ".csv"), row.names = FALSE)
}
####################
# path_out = 'C:\\Users\\54651\\Downloads\\consumers project'
# for (pk in array(1:1)) {
#   AMI_data <- GetAMIData(
#                           AccountName = "derconsumersami",
#                           Credential = Credential,
#                           PartitionKey = pk,
#                           RowKey = NULL,
#                           ColumnName = "AMIData",
#                           StartDate = lubridate::ymd("2019-01-01"),  # includes whole month's usage
#                           EndDate = lubridate::ymd("2019-12-01"),  # includes whole month's usage
#                           TablePrefix = "RawHourly",
#                           Uncompress = TRUE,
#                           Verbose = FALSE
#   )
#   write.csv(d, glue("part", pk, ".csv"), row.names = FALSE)
# }

