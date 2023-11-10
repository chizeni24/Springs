# Springs
Developed a comprehensive hydro-climatic analysis framework by fetching and processing data through REST API calls to USGS spring flow and NOAA climate repositories. The project involved deploying advanced ensemble algorithms to construct a predictive time series model, further refined using genetic algorithms for hyperparameter optimization. Shapley's additive explanations were integrated to ensure transparency and build trust in the model's predictions. Additionally, I engineered data pipelines to process and downscale global circulation models (GCM) data to specific locales. Utilizing Recurrent Neural Networks (RNN), I projected spring flow under various climate scenarios. All findings were meticulously documented and illustrated through interactive dashboards, and collaborative efforts were made to compose and finalize the manuscript.


Spring Flow projections for Comal and San Marcos Springs
Training and Test from 1960 -2009  and 2009 -2020 respectively
Prjection from 2020 to 2100

Each file contains 3 models Extra Tree,XGBoost and catboost(untuned and tuned)

At the begining of each file are the path you need to set in order to 
save all the results.

For Predicting Spring Flow using Explainable Artificial Intelligence Models.
