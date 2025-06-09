# Load necessary libraries
library(shiny)
library(shinythemes)
library(bslib)
library(readr)
library(haven)
library(DT)

# Source the helper functions
source("C:/Users/Admin/OneDrive/Desktop/Personal Projects/Apps/DMLSP/helpers.R")

# Install and load required packages
install_and_load_packages()

# Define UI for the Shiny app
ui <- fluidPage(theme=shinytheme("united"),
                
                headerPanel("Sean's DML App"),
                #Input Values
                sidebarPanel(
                  HTML("<h3>DML Options</h3>"),
                  
                  fileInput("data_file", label = "Choose a data file:", 
                              accept=c(".dta", ".csv")),
                  selectInput("outcome_var", "Outcome Variable:",
                              choices=NULL),
                  selectInput("treatment_var", "Treatment Variable:",
                              choices=NULL),
                  uiOutput("features_ui"),
                  
                  h3("Model Selection"), 
                  selectInput("model_type", "Select Model Type:",
                              choices = c("Elastic Net" = "enet",
                                          "Regression Trees" = "trees",
                                          "Random Forest" = "forest",
                                          "Boosted Trees" = "boosted")),
                  
                  actionButton("process_data", "Process Data"),
                  actionButton("PDLS", "Run Double Lasso Selection"),
                  actionButton("Use_PDLS", "Use Double Lasso Selection Features"),
                  actionButton("run_dml", "Run Double Machine Learning"),
                  ),
                # Main panel for displaying results
                mainPanel(
                  h3("Results"), 
                  DT::dataTableOutput("processed_table"),
                  verbatimTextOutput("pdls_features"),
                  verbatimTextOutput("summary_text"), 
                  tableOutput("results_table"), 
                  verbatimTextOutput("confints_text"),
                  verbatimTextOutput("estimates_text")
                )
              )

# Define server function
server <- function(input, output, session) {
  # Set the maximum upload size to 100MB
  options(shiny.maxRequestSize = 100*1024^2)
  
  data <- reactiveVal() 
  processed_data <- reactiveVal() 
  results <- reactiveVal() 
  pdls_features <- reactiveVal()

  observeEvent(input$data_file, { 
    req(input$data_file) 
    inFile <- input$data_file 
    
    if (grepl("\\.dta$", inFile$name)) { 
      uploaded_data <- haven::read_dta(inFile$datapath)
    } else if (grepl("\\.csv$", inFile$name)) {
      uploaded_data <- readr::read_csv(inFile$datapath)
    }
    data(uploaded_data) 
    
    updateSelectInput(inputId = "outcome_var", choices = names(uploaded_data)) 
    updateSelectInput(inputId = "treatment_var", choices = names(uploaded_data))
  })
  
  output$features_ui <- renderUI({ 
    req(data()) 
    tagList(
      h4("Select Features"), 
      selectInput("features", "", choices = names(data()), multiple = TRUE)
    )
  })
  
  observeEvent(input$process_data, { 
    req(data(), input$outcome_var, input$treatment_var, input$features) 
    
    processed_data(preprocess_data(data(), 
                                   input$outcome_var, 
                                   input$treatment_var, 
                                   input$features))
    
    showNotification("Data processed successfully!", type = "message")
  })
  
  output$processed_table <- DT::renderDataTable({
    req(processed_data())
    processed_data()
  }, options=list(scrollX=TRUE, scrollY="480px"))
  
  observeEvent(input$PDLS,{
    req(processed_data(), input$outcome_var, input$treatment_var, input$features)
    
    selected <- double_lasso_selection(processed_data(), 
                           input$outcome_var,
                           input$treatment_var,
                           input$features)
    pdls_features(selected)
    showNotification("Double Lasso Selection completed!", type="message")
  })
  
  output$pdls_features <- renderPrint({
    req(pdls_features())
    cat("Selected features by Double Lasso:\n")
    print(pdls_features())
    if(length(pdls_features())<2){
      cat("Warning: Post-Double Lasso Selection has selected less than two features.\n",
      "You cannot use Double Machine Learning with less than two Features. \n")
    }
  })
  
  observeEvent(input$Use_PDLS,{
    req(processed_data(), pdls_features(),input$outcome_var, input$treatment_var, input$features)
    updateSelectInput(
      session=session,
      inputId="features",
      selected = pdls_features()
    )    
    showNotification("Using Double Lasso Selection Features", type="message")
  })
  
  observeEvent(input$run_dml, {
    req(processed_data(), input$outcome_var, input$treatment_var, input$features, input$model_type)
    
    results(run_dml(processed_data(), 
                    input$outcome_var, 
                    input$treatment_var, 
                    input$features, 
                    input$model_type))
    
    showNotification("Double Machine Learning completed!", type = "message")
    
  })
  # Display DML model summary
  output$summary_text <- renderPrint({
    req(results())
    print(results())  # or summary(results()) if you have an S3 summary method
  })
  
  # Display results table (coef, confint, pval, se)
  output$results_table <- renderTable({
    req(results())
    data.frame(
      Estimate = results()$coef,
      Lower = results()$confint()[, 1],
      Upper = results()$confint()[, 2],
      P_value = results()$pval,
      Standard_Error = results()$se
    )
  }, rownames = TRUE)
  
  # Display confidence intervals as text
  output$confints_text <- renderPrint({
    req(results())
    results()$confint()
  })
  
  # Display coefficient estimates as text
  output$estimates_text <- renderPrint({
    req(results())
    results()$coef
  })
}

# Run the app
runApp(list(ui = ui, server = server))