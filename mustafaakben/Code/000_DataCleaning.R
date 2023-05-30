#########################################################################
#########################################################################
### ANCHOR Load the libraries

# Clear the global environment of all objects
rm(list = ls())
# Perform garbage collection, freeing up memory
gc()

# Set up a vector of package names
packages <- c("magrittr", "dplyr", "tidyverse", "stringr")

# Check each package: if it's not installed, install it, then load it
invisible(lapply(packages, function(pkg) {
  if (!pkg %in% installed.packages()) {
    # Install the package if it's not installed already
    install.packages(pkg)
  }
  # Load the package in the session
  library(pkg, character.only = TRUE)
}))


#########################################################################
#########################################################################
### ANCHOR Utility Functions

# Set the main directory as the current working directory
main_dir <- getwd()

# Function to set the working directory
set_dir <- function(dir, new_dir) {
    # Concatenate the directory paths and set the working directory
    setwd(file.path(dir, new_dir))
    cat("\n\nWorking Dir is set to:\n", getwd())
}

# Check if the operating system is Linux
is_linux <- Sys.info()[["sysname"]] == "Linux"

# Define the 'cls' function based on the OS type
if (is_linux) {
    cls <- function() system("clear")
} else {
    cls <- function() shell("cls")
}

# Function to detect a pattern in text and replace it with specified replacement
detect_and_replace <- function(text, pattern, replace = "", max_out = 2000) {
    # Detect the pattern
    cases <- grepl(pattern = pattern, x = text, ignore.case = TRUE, perl = TRUE)

    # Count the number of matches
    num_cases <- sum(cases)

    print(paste("Number of cases detected:", num_cases))

    # Replace the pattern
    text <- gsub(pattern = pattern, replacement = replace, x = text, ignore.case = TRUE, perl = TRUE)

    # Truncate the text to max_out characters
    text <- substr(text, 1, max_out)

    return(text)
}

# Function to apply 'detect_and_replace' function to multiple patterns
replace_patterns <- function(patterns_vector, input_text) {
    # Apply 'detect_and_replace' function for each pattern in patterns_vector
    for (pattern in patterns_vector) {
        input_text <- detect_and_replace(input_text, pattern)
    }
    return(input_text)
}

# Function to remove punctuation and digits from the text
delete_punct_and_digits <- function(text) {
    # Replace punctuation and digits with space
    text <- gsub(pattern = "[[:punct:]]", replacement = " ", x = text, ignore.case = TRUE, perl = TRUE)
    text <- gsub(pattern = "[[:digit:]]", replacement = " ", x = text, ignore.case = TRUE, perl = TRUE)

    # Replace multiple spaces with single space
    text <- gsub(pattern = "\\s+", replacement = " ", x = text, ignore.case = TRUE, perl = TRUE)

    # Remove leading and trailing white spaces
    text <- trimws(text)

    return(text)
}


cls()


#########################################################################
#########################################################################
### ANCHOR 02-Upload Data

# Read in the training data as a tibble

# NOTE: Change the data path

data <- read.csv("data/train_pub.csv")  %>% as_tibble()
# Add a new column 'test_train' to differentiate between test and train data
data$test_train <- "TRAIN"

# Read in the first test data set as a tibble
test_data <- read.csv("data/dev_pub.csv")  %>% as_tibble()
# Add a new column 'test_train' to differentiate between test and train data
test_data$test_train <- "TEST"

# Read in the second test data set as a tibble
test_data2 <- read.csv("data/test_pub.csv")  %>% as_tibble()
# Add a new column 'test_train' to differentiate between test and train data
test_data2$test_train <- "TEST"

# Store the response ids of the second test data for future use
TEST_IDS <- test_data2$response_id
# Merge train data and test data into one dataset
data <- rbind(data, test_data, test_data2) %>% as_tibble()

# Add an ID column to the data
data$ID <- seq_len(nrow(data))



#########################################################################
#########################################################################
### ANCHOR Email Cleaning from the columns


encoded_data <- data %>% tibble()

# text_exercise_4
pattern_vector <- c("(From: Note).*(Attachment)")
encoded_data$text_exercise_4 <- replace_patterns(pattern_vector, encoded_data$text_exercise_4)

# text_exercise_5
pattern_vector <- c("(From: Hurdle).*(Attachment)")
encoded_data$text_exercise_5 <- replace_patterns(pattern_vector, encoded_data$text_exercise_5)


# text_exercise_6
pattern_vector <- c("(From: Paxton).*(Trim Install)","(Subject : Debby).*(Trim Install)","(\\$0)")
encoded_data$text_exercise_6 <- replace_patterns(pattern_vector, encoded_data$text_exercise_6)


# text_exercise_7
pattern_vector <- c("(From: Paxton).*(Trim Install)","(Subject : Kirkland Plant).*(Trim Install)","(Im hearing some).*(Trim Install)","(\\$0)")
encoded_data$text_exercise_7 <- replace_patterns(pattern_vector, encoded_data$text_exercise_7)

# text_exercise_8
pattern_vector <- c("(From: Slade).*(Slade)")
encoded_data$text_exercise_8 <- replace_patterns(pattern_vector, encoded_data$text_exercise_8)


# text_exercise_9
pattern_vector <- c(
    "(From: Norman).*(Soundproof Solutions)", 
    "(Professional Conduct).*(Soundproof Solutions)",
    "(Professional Conduct).*(Norman Manager)", 
    "(Professional Conduct).*(NormanManager)", 
    "(You should know that salespeople).*(Manager)",
    "(You should know that).*(You can watch me work anytime)", "(\\$0)"
)
encoded_data$text_exercise_9 <- replace_patterns(pattern_vector, encoded_data$text_exercise_9)


# text_exercise_10
pattern_vector <- c("(From: Rothc).*( productivity than the corporate standard.)")
encoded_data$text_exercise_10 <- replace_patterns(pattern_vector, encoded_data$text_exercise_10)


# text_exercise_11
pattern_vector <- c(
    "(From: Foster).*(Administrative Assistant)",
    "(Subject : SEQUENCE Talk).*(Administrative Assistant)", 
    "(I am planning the first SEQUENCE Talk).*(Administrative Assistant)", 
    "(\\$0)"
)
encoded_data$text_exercise_11 <- replace_patterns(pattern_vector, encoded_data$text_exercise_11)


# text_exercise_12
pattern_vector <- c(
    "(From: Lake).*(ProcessesAttachment)",
    "(From: Lake).*(Attachment)",
    "(Subject : Team Focus).*(Attachment)",
    "(I wanted to draw your attention to a troubling).*(ProcessesAttachment)",
    "(Please review the attached).*(Attachment)","(\\$0)")
encoded_data$text_exercise_12 <- replace_patterns(pattern_vector, encoded_data$text_exercise_12)


# text_exercise_13
pattern_vector <- c("(From: Roth).*(Attachment)","(Strength Course Completion ).*(Attachment)")
encoded_data$text_exercise_13 <- replace_patterns(pattern_vector, encoded_data$text_exercise_13)


# text_exercise_14
pattern_vector <- c(
    "(From: Paxton).*(Trim Install)",
    "(Subject : Turntable Proposal).*(Trim Install)",
    "(We are pretty close to the end).*(PaxtonLead)",
    "(We are pretty close to the end).*(Paxton Lead)","(\\$0)")
encoded_data$text_exercise_14 <- replace_patterns(pattern_vector, encoded_data$text_exercise_14)


# text_exercise_15
pattern_vector <- c(
    "(Upgrades to Robot Software).*(TecniovManager)",
    "(Upgrades to Robot Software).*(Tecniov Manager)",
    "(\\$0)")
encoded_data$text_exercise_15 <- replace_patterns(pattern_vector, encoded_data$text_exercise_15)


# text_exercise_16
pattern_vector <- c("(\\$0)")
encoded_data$text_exercise_16 <- replace_patterns(pattern_vector, encoded_data$text_exercise_16)


# text_exercise_17
pattern_vector <- c(
    "(From:).*(.*Attachment)",
    "(Victory lunch).*(.*Attachment)",
    "(\\$0)")

encoded_data$text_exercise_17 <- replace_patterns(pattern_vector, encoded_data$text_exercise_17) 


# text_exercise_18
pattern_vector <- c("(From:).*(.*SladeCoordinator)")
encoded_data$text_exercise_18 <- replace_patterns(pattern_vector, encoded_data$text_exercise_18)


# text_exercise_19
pattern_vector <- c(
    "(From:).*(.*Paul Bern Predecessor)",
    "(From: Daisha).*(Plant Security)",
    "(From: Chun).*(Security Guard)", "(\\$0)"
)
encoded_data$text_exercise_19 <- replace_patterns(pattern_vector, encoded_data$text_exercise_19)






###############################################################################
#### ANCHOR Extract Extra Information text_exercise_final

# Set the variable 'text' to the 'text_exercise_final' column from the encoded_data dataframe
text <- encoded_data$text_exercise_final



# Define the pattern to filter questions based on the simulation game
pattern <- "(Did you handle each challenge in the order you received it, or did you handle them another way)"

# Apply the filter and get the indices of the matching strings
filter <- grepl(pattern, x = text, ignore.case = TRUE, perl = TRUE)  %>% which

# Print the length of the filter
cat("\n\nLength of filtered datapoints:", length(filter),'\n\n')

# Get the row indices of the filtered items
row_idx <- encoded_data$ID[filter]

# Create a temporary copy of the encoded data
temp_df <- encoded_data

# Filter the text
text <- text_original <- text[filter]

# Define various patterns and their replacements to clean the text
patterns_subs <- list(
  c("SoundproofSolutions", "Soundproof Solutions"),
  c("ofSoundproof Solutions", "of Soundproof Solutions"),
  c("Didyou", "Did you"),
  c("themanother", "them another"),
  c("(  )", "\n\n"),
  c("(Did you handle each challenge in the order you received it, or did you handle them another way)", "\n\n### Q-ONE: \n\n"),
  c("(Please list the major categories of issues or problems facing Final Assembly in the Bridgeport facility of Soundproof Solutions)", "\n\n### Q-TWO: \n\n"),
  c("(Describe why each entry in Question 2 is an issue or problem)", "\n\n### Q-THREE: \n\n"),
  c("(What were the three most important and the three least important e-mails to handle)", "\n\n### Q-FOUR: \n\n"),
  c("(Why did you feel that the e-mails you rated as most important were critical)", "\n\n### Q-FIVE: \n\n"),
  c("(Todays assessment provided you with background information).*(Policies related to Message Y, etc\\.\\)\\.)", "\n\n### Q-SIX: \n\n"),
  c("(From:).*(Administrator Sent)", "####"),
  c("(Please describe your approach)", ""),
  c("(Original Message)", ""),
  c("(This e-mail and the one that follows)[\\W\\w]*(responses in the form of a reply e-mail)", "####"),
  c("(Following are some of)[\\W\\w]*(form of a reply e-mail.)", "####")
)

# Apply the 'detect_and_replace' function for each pattern-sub pair
for(pattern_sub in patterns_subs) {
  text <- detect_and_replace(text, pattern = pattern_sub[1], replace = pattern_sub[2], max_out = 102000)
}

# Store the mid-stage text
mid_text <- text


# Split the 'text' by '#' 
text_split <- stringr::str_split(text, pattern = "#")

# Filter out short strings from each split text
for (each in 1:length(text_split)) {
    selected <- lapply(text_split[each], FUN = function(X) nchar(X) > 10) %>% unlist()
    text_split[[each]] <- text_split[[each]][selected]
}

# Define questions for the columns of the new dataframe
questions <- c("Q-ONE", "Q-TWO", "Q-THREE", "Q-FOUR", "Q-FIVE", "Q-SIX", "Q-OTHERS")

# Initialize a dataframe to hold question data
question_data <- matrix(NA, nrow = length(text_split), ncol = length(questions)) %>% data.frame()
names(question_data) <- questions

# Iterate through each split text and assign the parts to the corresponding columns
for (j in 1:length(text_split)) {
    if (j %% 120 == 0) cat("\nITERATION : ", j, "\n")

    temp <- text_split[[j]]
    max <- 1:length(temp)
    in_ <- c()

    for (i in 1:ncol(question_data)) {
        idx_bool <- lapply(temp, FUN = function(X) grepl(pattern = questions[i], x = X, ignore.case = TRUE, perl = TRUE)) %>% unlist()
        ids <- which(idx_bool)

        tt <- paste(temp[ids], collapse = "\n")
        tt <- stringr::str_squish(tt)
        if (nchar(tt) < 15) tt <- NA
        question_data[j, i] <- tt

        in_ <- c(ids, in_)
    }

    ids <- which(!max %in% in_)
    tt <- paste(temp[ids], collapse = "\n")
    tt <- stringr::str_squish(tt)
    question_data[j, 7] <- tt
    in_ <- c()
}

# Display a sample of the question_data dataframe
cls()
question_data %>% sample_n(1) %>% head(2)

# Add 'ID' column to the question_data dataframe
question_data$ID <- row_idx

# Merge encoded_data and question_data dataframes
encoded_data <- full_join(encoded_data, question_data)

# Remove unwanted columns from the dataframe
encoded_data <- encoded_data[!startsWith(names(encoded_data), "HEAD")]
encoded_data <- encoded_data[setdiff(names(encoded_data), "others")]

# Change column names by replacing '-' with '.'
change_names <- names(encoded_data)[startsWith(names(encoded_data),"Q")]
change_names <- gsub("-", ".", change_names)
names(encoded_data)[startsWith(names(encoded_data),"Q")] <- change_names



#########################################################################
#########################################################################
### ANCHOR Extract the Second Simulation Information from text_exercise_final


# Calculate the sum of NA values in columns from 'Q.ONE' to 'Q.OTHERS' for each row
N_na <- encoded_data %>% select(`Q.ONE`:`Q.OTHERS`) %>% is.na() %>% rowSums()

table(N_na)

# Add the sum of NA values as a new column 'N_na' to 'encoded_data'
encoded_data$N_na <- N_na

# Create two dataframes, one with rows where 'text_exercise_final' is not NA and 'N_na' equals 7, 
# and the other with the rest of the rows
filter <- !is.na(encoded_data$text_exercise_final) & encoded_data$N_na == 7
split_data1 <- encoded_data %>% filter(filter)
split_data2 <- encoded_data %>% filter(!filter)

# Select 'text_exercise_final' from 'split_data1'
text <- split_data1$text_exercise_final
length(text)


# Create a list of patterns and replacements
replacements <- list(
  "SoundproofSolutions" = "Soundproof Solutions",
  "ofSoundproof Solutions" = "of Soundproof Solutions",
  "Didyou" = "Did you",
  "themanother" = "them another",
  "(  )" = "\n\n"
)

# Loop over the list and replace each pattern with its corresponding replacement
for (i in seq_along(replacements)) {
  text <- detect_and_replace(text, pattern = names(replacements[i]), replace = replacements[i], max_out = 102000)
}



# Create a list of patterns and replacements
replacements <- list(
  "Please list the major categories of issues or problems facing Customer Service Team 5 of Soundproof Solutions. Why is each an issue or a problem" = "\n\n\\[\\[SEP\\]\\] TYPE_2_ONE: ",
  "(What were the three most important messages).* (Why?)" = "\n\n\\[\\[SEP\\]\\] TYPE_2_TWO: ",
  "(List specific examples \\(if any\\)).*(but in other message\\(s\\))" = "\n\n\\[\\[SEP\\]\\] TYPE_2_THREE: ",
  "(Describe what additional information you would).*(messages and challenges presented to you today)" = "\n\n\\[\\[SEP\\]\\] TYPE_2_FOUR: "
)

# Loop over the list and replace each pattern with its corresponding replacement
for (i in seq_along(replacements)) {
  text <- gsub(text, pattern = names(replacements[i]), replacement = replacements[i], ignore.case = TRUE, perl = TRUE)
}

# Create a list of patterns and replacements
patterns_and_replacements <- list(
  "(From:).*(a list of messages you received today is shown after question 4)" = "\n\n\\[\\[SEP\\]\\]",
  "(The messages you received today).*(An unhappy customer demands reinstallation and a meeting)" = "\n\n\\[\\[SEP\\]\\]"
)

# Loop over the list and replace each pattern with its corresponding replacement
for (pattern in names(patterns_and_replacements)) {
  replacement = patterns_and_replacements[[pattern]]
  text <- gsub(text, pattern = regex(pattern, dotall = TRUE, multiline = T, ignore_case = TRUE), 
               replacement = replacement)
}


# Clear the console
cls()
# Print a sample from the 'text'
cat(sample(text, 1))


# Clear Console
cls()
# Define a split pattern
split_pattern <- "\\[\\[SEP\\]\\]"

# Split the text into a list of strings based on the split pattern
text_split <- stringr::str_split(string = text, pattern = split_pattern)

# Remove strings that have fewer than 2 characters
text_split <- lapply(text_split, function(x) x[nchar(x) > 2])

# Define question identifiers
questions <- c("TYPE_2_ONE", "TYPE_2_TWO", "TYPE_2_THREE", "TYPE_2_FOUR")

# Initialize a dataframe to store the responses to each question
question_data <- data.frame(matrix(NA, nrow = length(text_split), ncol = length(questions)))
names(question_data) <- questions

# Iterate over the list of split text and match responses to questions
for (j in seq_along(text_split)) {
  
    # Status update
    if (j %% 120 == 0) cat("\nITERATION : ", j, "\n")

    # Temporary storage for each list of split text
    temp <- text_split[[j]]

    # For each question, extract the matching response and store in question_data
    for (i in seq_along(questions)) {
      
        # Identify index of response that matches the question
        ids <- grep(pattern = questions[i], x = temp, ignore.case = TRUE, perl = TRUE)

        # Collapse multiple responses (if any) into a single string
        tt <- paste(temp[ids], collapse = " \n")
      
        # If the string has fewer than 3 characters, set to NA
        if (nchar(tt) < 3) tt <- NA

        # Store the response in the question_data dataframe
        question_data[j, i] <- tt
    }
    
    # Handle the remaining responses that do not match any of the questions
    ids <- setdiff(seq_along(temp), ids)
    tt <- paste(temp[ids], collapse = "\n")
    question_data[j, "TYPE_2_OTHER"] <- tt
}

# Convert dataframes to tibbles for better handling
question_data <- question_data %>% as_tibble()
split_data1 <- data.frame(split_data1, question_data) %>% as_tibble()

# Merge the split data
encoded_data <- full_join(split_data2, split_data1)

# Arrange the merged data by ID
encoded_data <- encoded_data  %>% arrange(ID)

# List of columns to clean up
columns_to_clean <- c("TYPE_2_ONE", "TYPE_2_TWO", "TYPE_2_THREE", "TYPE_2_FOUR")

# Use mutate_at to apply the stringr::str_squish function to multiple columns
encoded_data <- encoded_data %>% 
  dplyr::mutate_at(vars(columns_to_clean), stringr::str_squish)


# cls()
# sample(encoded_data$TYPE_2_ONE[!is.na(encoded_data$TYPE_2_ONE)],1)  %>% cat


#########################################################################
#########################################################################
### ANCHOR Final PreProcessing 

# Defining the question tags
question_tags <- c(
  "Q-ONE:", "Q-TWO:", "Q-THREE:", "Q-FOUR:", "Q-FIVE:", "Q-SIX:", "Q-OTHERS:", 
  "TYPE_2_ONE:", "TYPE_2_TWO:", "TYPE_2_THREE:", "TYPE_2_FOUR:"
)

# Selecting the relevant data
text_data <- encoded_data  %>% dplyr::select(Q.ONE:TYPE_2_OTHER)

# Removing the question tags from each column
text_data <- lapply(text_data, function(text_col) {
  for (question_tag in question_tags) {
    text_col <- gsub(pattern = question_tag, replacement = "", x = text_col, ignore.case = TRUE, perl = TRUE)
  }
  stringr::str_squish(text_col)
})

# Converting the list back to a data frame
text_data <- as.data.frame(text_data)

# Applying the delete_punct_and_digits function to all the columns
text_data <- lapply(text_data, delete_punct_and_digits) %>% as.data.frame()

# Updating the original data frame
encoded_data[names(text_data)] <- text_data

# Cleaning email "\" character
email_cols <- encoded_data  %>% dplyr::select(text_exercise_4:text_exercise_19) %>% colnames()

encoded_data[email_cols] <- lapply(encoded_data[email_cols], function(x) {
  stringr::str_replace_all(x,'\"',"")
})

cls()
sample(encoded_data$TYPE_2_ONE[!is.na(encoded_data$TYPE_2_ONE)],1)  %>% cat


#########################################################################
#########################################################################
### ANCHOR Encode Missing Patterns

# Select and scale the relevant columns
scaled_data <- encoded_data %>%
  dplyr::select(text_exercise_4:text_exercise_19) %>%
  is.na() %>%
  as.data.frame() %>%
  lapply(scale) %>%
  as.data.frame()

# Perform kmeans clustering
kmeans_model <- kmeans(x = scaled_data, centers = 3, iter.max = 2000, nstart = 20000)

# Add the cluster results to the original data frame
encoded_data$TYPE2 <- kmeans_model$cluster

# Display the number of instances in each cluster
table(encoded_data$TYPE2)



## Extract Player Names from the data
encoded_data$playerName <- NA

player_names <- c("Jamie Pace", "Cory Manning", "Cary Stevens")
player_names_in_email <- c("Pace, Jamie ", "Manning, Cory", "Stevens, Cary")

for (each_player in seq_len(length(player_names))) {
    pattern <- paste("To: ", player_names_in_email[each_player], sep = "")
    encoded_data$playerName[grepl(pattern = pattern, x = encoded_data$text_exercise_final, ignore.case = TRUE, perl = TRUE)] <- player_names[each_player]
}

## For missing player names search all the emails
full_player <- encoded_data  %>% filter(!is.na(playerName)) 
missing_player <- encoded_data  %>% filter(is.na(playerName)) 

## Extract the player name from the missing player from Email Data
emails <- missing_player  %>% select(text_exercise_4:text_exercise_19)
# List of names
names <- c("Jamie", "cory", "cary")
# Function to count occurrences
count_occurrences <- function(name, emails) {
    apply(emails, 1, function(X) sum(agrepl(pattern = name, x = X, max.distance = 0.1, ignore.case = TRUE, useBytes = TRUE)))
}

# Create the data frame
missing_player_n <- data.frame(sapply(names, count_occurrences, emails = emails))
names(missing_player_n) <- paste0("n_", tolower(names))
# Based on the max column, assgin the player name
missing_player$playerName <- player_names[apply(missing_player_n, 1, which.max)]

## Combine the data
encoded_data <- rbind(full_player, missing_player)  

## Cros-table TYPE2 in encoded and the playernames
table(encoded_data$TYPE2, encoded_data$playerName)


## Change the all player names where Cary Stevens is loaded on the cluster in Type2
cross_table <- table(encoded_data$TYPE2, encoded_data$playerName)
cary_stevens_loading <- which.max(cross_table[,colnames(cross_table) == "Cary Stevens"])

filter1 <- encoded_data$TYPE2 == cary_stevens_loading 
encoded_data$playerName[filter1] <- "Cary Stevens"

filter2 <- encoded_data$playerName == "Cary Stevens" & encoded_data$TYPE2 != cary_stevens_loading
encoded_data$playerName[filter2] <- "Cory Manning"

# Recheck loadings
table(encoded_data$TYPE2, encoded_data$playerName)



#########################################################################
#########################################################################
### ANCHOR Prepare for writing - UTF 8

names(encoded_data)

# Change the names of Q.ONE to TYPE_1_ONE and so on
names(encoded_data) <- gsub(pattern = "Q\\.", replacement = "TYPE_1_", x = names(encoded_data), ignore.case = TRUE, perl = TRUE)


extracted_information <- c("TYPE_1_ONE", "TYPE_1_TWO", "TYPE_1_THREE", "TYPE_1_FOUR", "TYPE_1_FIVE", "TYPE_1_SIX", "TYPE_2_ONE", "TYPE_2_TWO", "TYPE_2_THREE", "TYPE_2_FOUR")
text_data <- encoded_data[extracted_information]

for (each in seq_len(ncol(text_data))) {
  text <- text_data %>% pull(each)
  text <- gsub(pattern = "\"", x = text, replacement = "", ignore.case = TRUE)
  text <- stringr::str_squish(text)
  text_data[, each] <- text
}

encoded_data[names(text_data)]  <- text_data


# Add a column for additional data that shows the test_data2 or test_data
encoded_data$additional_data <- (encoded_data$response_id %in% test_data2$response_id)*1

## Reoder the dataframe
column_order <- c(
    ## Meta Information
    "test_train",
    "additional_data",
    "TYPE2",
    "playerName",
    "text_exercise_final",
    "ID",
    "response_id",
    "playerName",
    # Ratings
    "rating_chooses_appropriate_action", "rating_commits_to_action",
    "rating_gathers_information", "rating_identifies_issues_opportunities",
    "rating_interprets_information", "rating_involves_others",
    "rating_decision_making_final_score",
    # Text Data
    "text_exercise_4", "text_exercise_5", "text_exercise_6", "text_exercise_7",
    "text_exercise_8", "text_exercise_9", "text_exercise_10", "text_exercise_11",
    "text_exercise_12", "text_exercise_13", "text_exercise_14", "text_exercise_15",
    "text_exercise_16", "text_exercise_17", "text_exercise_18", "text_exercise_19",
    # Extracted Information
    "TYPE_1_ONE", "TYPE_1_TWO", "TYPE_1_THREE", "TYPE_1_FOUR", "TYPE_1_FIVE", "TYPE_1_SIX",
    "TYPE_1_OTHERS",
    "TYPE_2_ONE", "TYPE_2_TWO", "TYPE_2_THREE", "TYPE_2_FOUR", "TYPE_2_OTHER"
)

## ReOrder Columns
encoded_data <- encoded_data[column_order]

## Create a data folder if it does not exist
if (!dir.exists("data")) {
    dir.create("data")
}

## Write the data with UTF-8 and NA is empty coded as ""
write.csv(encoded_data, file = "data/FullData.csv", row.names = FALSE, na = "", fileEncoding = "UTF-8")


## The rest of the analysis is carried out with the Python Script
## Please open the 001_EnsembleModel.py

## END OF SCRIPT