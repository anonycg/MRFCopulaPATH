# NOTE: To load data, you must download both the extract's data and the DDI
# and also set the working directory to the folder with these files (or change the path below).

#if (!require("ipumsr")) stop("Reading IPUMS data into R requires the ipumsr package. It can be installed using the following command: install.packages('ipumsr')")


library('ipumsr')
library(dplyr)


#ddi <- read_ipums_ddi("usa_00009.xml")  #ACS 2019
ddi <- read_ipums_ddi("usa_00010.xml")   #2018-2022, ACS 5-year
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
#data_Juri_copy$STATEFIP <- as.character(data_Juri_copy$STATEFIP)
data_Juri_copy <- data_Juri_copy %>%
  mutate(FIPS_just_state = case_when(
    1 <= STATEFIP & STATEFIP <= 9 ~ paste0("0", as.character(STATEFIP)),
    TRUE ~ as.character(STATEFIP)
  ))


#data_Juri_copy$FIPS <- paste0(data_Juri_copy$STATEFIP, data_Juri_copy$FIPS_without_state)
data_Juri_copy$FIPS <- paste0(data_Juri_copy$FIPS_just_state, data_Juri_copy$FIPS_without_state)


data_Juri_copy <- data_Juri_copy %>%
  filter(AGE >= 18)
#POVERTY ===================================

# Persons below 100% poverty estimate
data_Juri_copy <- data_Juri_copy %>% 
  mutate(SDH1_poverty = ifelse(POVERTY <= 100, 1, 0))

# 000 = N/A, deleting those rows
data_Juri_copy1 <- data_Juri_copy %>%
  filter(POVERTY != 0)

# Applying the 'PERWt' weight for each row, to be representative of the national level
# Calculating the percentage of people with social condition for each jurisdiction
weighted_marginal <- data_Juri_copy1 %>%
  group_by(FIPS,SDH1_poverty) %>%
  summarize(sum_perwt = sum(PERWT)) %>%
  group_by(FIPS) %>%
  mutate(sum_sum_perwt = sum(sum_perwt),
         percentage = sum_perwt / sum(sum_perwt))

write_xlsx(weighted_marginal, "marginal_poverty.xlsx")



# UNEMPLOYED ==========================


# Civilian (age 18+) unemployed estimate
data_Juri_copy <- data_Juri_copy %>%
  filter(AGE >= 18)
data_Juri_copy <- data_Juri_copy %>%
  mutate(SDH2_employment = ifelse(EMPSTATD %in% c(20, 21, 22), 1, 0))

value_counts_emp <- table(data_Juri_copy$SDH2_employment)
#data_Juri_copy <- data_Juri_copy %>%
#  mutate(SDH2_employment = ifelse(EMPSTATD == 20, 1, 0))   

### data cleaning - deleting the following rows:
# 00 = N/A; 14,15 = armed force; 30 = not in labor force
data_Juri_copy2 <- data_Juri_copy %>%
  filter(!(EMPSTATD %in% c("0", "15", "14",'13', "30")))
data_Juri_copy2 <- data_Juri_copy2 %>%
  filter(EMPSTAT != 3) #not in labor force

# Applying the 'PERWt' weight for each row, to be representative of the national level
# Calculating the percentage of people with social condition for each jurisdiction
weighted_marginal <- data_Juri_copy2 %>%
  group_by(FIPS,SDH2_employment) %>%
  summarize(sum_perwt = sum(PERWT)) %>%
  group_by(FIPS) %>%
  mutate(sum_sum_perwt = sum(sum_perwt),
         percentage = sum_perwt / sum(sum_perwt))

write_xlsx(weighted_marginal, "marginal_employment.xlsx")


#Insurance ================================


# % of Uninsured in the total civilian non-institutionalized population estimate
data_Juri_copy <- data_Juri_copy %>%
  mutate(SDH3_insurance = ifelse(HCOVANY == 1, 1, 0))

weighted_marginal <- data_Juri_copy %>%
  group_by(FIPS,SDH3_insurance) %>%
  summarize(sum_perwt = sum(PERWT)) %>%
  group_by(FIPS) %>%
  mutate(sum_sum_perwt = sum(sum_perwt),
         percentage = sum_perwt / sum(sum_perwt))

write_xlsx(weighted_marginal, "marginal_insurance.xlsx")



#Education ==============================================

# % of Persons (age 25+) with no high school diploma estimate

data_Juri_copy3 <- data_Juri_copy %>%
  filter(AGE >= 25)

# data cleaning: 

# N/A
data_Juri_copy3 <- data_Juri_copy3 %>%
  filter(!(EDUCD == 1 & EDUC == 0))


# for missing data EDUCD = 999
data_Juri_copy3 <- data_Juri_copy3 %>%
  filter(EDUCD != 999)

data_Juri_copy3 <- data_Juri_copy3 %>% 
  mutate(SDH4_education = ifelse(EDUCD <= 61, 1, 0))


weighted_marginal <- data_Juri_copy3 %>%
  group_by(FIPS,SDH4_education) %>%
  summarize(sum_perwt = sum(PERWT)) %>%
  group_by(FIPS) %>%
  mutate(sum_sum_perwt = sum(sum_perwt),
         percentage = sum_perwt / sum(sum_perwt))

write_xlsx(weighted_marginal, "marginal_education.xlsx")


