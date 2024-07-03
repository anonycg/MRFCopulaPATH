# NOTE: To load data, you must download both the extract's data and the DDI
# and also set the working directory to the folder with these files (or change the path below).

#if (!require("ipumsr")) stop("Reading IPUMS data into R requires the ipumsr package. It can be installed using the following command: install.packages('ipumsr')")


library('ipumsr')
library(dplyr)

### loading data
ddi <- read_ipums_ddi("usa_00009.xml")   #ACS 2019
data_Juri <- read_ipums_micro(ddi)


### 000 = County not identifiable from public-use data (1950-onward)
data_Juri_copy <- data_Juri %>%
  filter(COUNTYFIP != 0)

### 99 = State not identified
data_Juri_copy <- data_Juri_copy %>%
  filter(STATEFIP != 99)

### convert to text and to 3 digit
data_Juri_copy <- data_Juri_copy %>%
  mutate(FIPS_without_state = case_when(
    1 <= COUNTYFIP & COUNTYFIP <= 9 ~ paste0("00", as.character(COUNTYFIP)),
    10 <= COUNTYFIP & COUNTYFIP <= 99 ~ paste0("0", as.character(COUNTYFIP)),
    TRUE ~ as.character(COUNTYFIP)
  ))

### convert state prefix to text
data_Juri_copy <- data_Juri_copy %>%
  mutate(FIPS_just_state = case_when(
    1 <= STATEFIP & STATEFIP <= 9 ~ paste0("0", as.character(STATEFIP)),
    TRUE ~ as.character(STATEFIP)
  ))
data_Juri_copy$FIPS <- paste0(data_Juri_copy$FIPS_just_state, data_Juri_copy$FIPS_without_state)


# create binary column for 4 IPUMS variables based on SVI definitions. 
# So each individual has either 0 for good status, or 1 for bad status.

#POVERTY ===================================
# Persons below 150% poverty estimate
data_Juri_copy <- data_Juri_copy %>% 
  mutate(SDH1_poverty = ifelse(POVERTY <= 150, 1, 0))

# 000 = N/A, deleting those rows
data_Juri_copy <- data_Juri_copy %>%
  filter(POVERTY != 0)

# UNEMPLOYED ==============================
# Civilian (age 16+) unemployed estimate
data_Juri_copy <- data_Juri_copy %>%
  filter(AGE >= 16)
data_Juri_copy <- data_Juri_copy %>%
  mutate(SDH2_employment = ifelse(EMPSTATD %in% c(20, 21, 22), 1, 0))


### data cleaning - deleting the following rows:
# 00 = N/A; 14,15 = armed force; 30 = not in labor force
data_Juri_copy <- data_Juri_copy %>%
  filter(!(EMPSTATD %in% c("0", "15", "14",'13', "30")))
data_Juri_copy <- data_Juri_copy %>%
  filter(EMPSTAT != 3) #not in labor force


#Insurance ================================
# % of Uninsured in the total civilian non-institutionalized population estimate
data_Juri_copy <- data_Juri_copy %>%
  mutate(SDH3_insurance = ifelse(HCOVANY == 1, 1, 0))

#Education ================================
# % of Persons (age 25+) with no high school diploma estimate
data_Juri_copy <- data_Juri_copy %>%
  filter(AGE >= 25)

# data cleaning: 
# N/A
data_Juri_copy <- data_Juri_copy %>%
  filter(!(EDUCD == 1 & EDUC == 0))

# for missing data EDUCD = 999
data_Juri_copy <- data_Juri_copy %>%
  filter(EDUCD != 999)

data_Juri_copy <- data_Juri_copy %>% 
  mutate(SDH4_education = ifelse(EDUCD <= 61, 1, 0))



###calculating marginals for each jurisdiction ######## count version just to check

marginal_poverty_c <- data_Juri_copy %>%
  group_by(FIPS) %>%
  summarize(marginal_poverty_c = sum(SDH1_poverty == 1))

marginal_employment_c <- data_Juri_copy %>%
  group_by(FIPS) %>%
  summarize(marginal_employment_c = sum(SDH2_employment == 1))

marginal_insurance_c <- data_Juri_copy %>%
  group_by(FIPS) %>%
  summarize(marginal_insurance_c = sum(SDH3_insurance == 1))

marginal_education_c <- data_Juri_copy %>%
  group_by(FIPS) %>%
  summarize(marginal_education_c = sum(SDH4_education == 1))



# calculating marginals (percentages) for each jurisdiction ########

marginal_poverty <- data_Juri_copy %>%
  group_by(FIPS) %>%
  summarize(marginal_poverty = sum(SDH1_poverty == 1) / n())

marginal_employment <- data_Juri_copy %>%
  group_by(FIPS) %>%
  summarize(marginal_employment = sum(SDH2_employment == 1) / n())

marginal_insurance <- data_Juri_copy %>%
  group_by(FIPS) %>%
  summarize(marginal_insurance = sum(SDH3_insurance == 1) / n())

marginal_education <- data_Juri_copy %>%
  group_by(FIPS) %>%
  summarize(marginal_education = sum(SDH4_education == 1) / n())





##### calculating the pairwise joint distribution: Pr(poverty, employment)
Pairwise_Joint_distribution <- data_Juri_copy %>%
  group_by(FIPS,SDH1_poverty, SDH2_employment ) %>%
  summarize(count = n()) %>%
  group_by(FIPS) %>%
  mutate(percentage = count / sum(count))

split_pairwise <- split(Pairwise_Joint_distribution, Pairwise_Joint_distribution$FIPS)

### list of jurisdiction that have full 4 combinations 
pairwise_combinations <- bind_rows(split_pairwise) %>%
  group_by(FIPS) %>%
  summarise(unique_combinations = n()) %>%
  filter(unique_combinations == 4)

#####
## list similar to split but only for FIPS codes with full 4 combinations
filtered_split_data <- list()  # Initialize an empty list to store the filtered data

for (i in seq_along(split_pairwise)) {
  fips_code <- names(split_pairwise)[i]  # Get the FIPS code
  if (fips_code %in% pairwise_combinations$FIPS) {
    filtered_split_data[[fips_code]] <- split_pairwise[[fips_code]]  # Keep the data for the FIPS code
  }
}

### converty to rows 
combined_data_test <- do.call(rbind, filtered_split_data)

required_data <- select(combined_data_test, FIPS, percentage)

library(tidyr)

reshaped_test <- required_data %>%
  mutate(row_number = ave(FIPS, FIPS, FUN = seq_along)) %>%
  spread(row_number, percentage)

library(openxlsx)

# Specify the file path for the Excel file
file_path <- "C:/Users/"

# Write the dataframe to an Excel file
write.xlsx(reshaped_test, file_path, rowNames = FALSE)
