- Archive Contents
This archive contains the deliverables from the extended simulation comparing federated learning algorithms and encryption schemes on the HAR dataset.

- Files and Directories

  
* enhanced_cfl_simulation.py:**  The main Python script used to execute the extended simulations. It includes implementations (with encountered issues) of FedAvg, FedMed, FedProx, FedOpt, CFL-KBS, encryption management, and metric collection.

* prepare_har_data.py:** The Python script used to download, extract, and preprocess the UCI HAR dataset, partitioning it for 30 federated clients.

Important Notes
-Most simulations (FedAvg, FedMed, FedOpt, CFL-KBS) failed due to technical issues. As a result, the results and visualizations primarily focus on the comparison between FedProx+ECC and FedProx+PE.

-The enhanced_cfl_simulation.py script is provided as is, including the implementations that caused errors, for reference and potential debugging purposes.


