import matplotlib.pyplot as plt
import numpy as np
from neptune import Run


def plotMetric(runID=None,metricName=None,setupIDs=None,trials=1,ylims=[0,1]):
  plt.figure(figsize=(40,10))
  
  # Variables for Neptune Authentication
  project="dimitri-kachler-workspace/sanity-MNIST"
  api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNWQxNDllOS04OGY1LTRjM2EtYTczZi0xNWI0NTRmZTA1OTEifQ=="
  seeRun = Run(api_token=api_token,project=project, with_id=runID, mode="read-only")
  
  # Building the iterations array
  if metricName != "loss":
    iterations = [i for i in range(len(seeRun[f"trials/0/{setupIDs[0]}/{metricName}"].fetch_values()["value"]))]
  else:
    iterations = [i for i in range(len( seeRun[f"trials/0/{setupIDs[0]}/loss"].fetch_values()["value"] )) ]

  # All of the methods in the setups file have their own trial values
  for method in setupIDs:
    methodValues = []
    for trial in range(trials):
      trialValues = seeRun[f"trials/{trial}/{method}/{metricName}"].fetch_values()["value"][:len(iterations)]
      methodValues.append(np.array(trialValues))

    #Finding the mean and standard error (see if along one dimension)
    methodMean = np.mean(methodValues,axis=0)
    methodSTD = np.sqrt(np.std(methodValues)) / 10


    plt.plot(iterations, methodMean,label=method)
    #plt.errorbar(iterations, methodMean, yerr = methodSTD)


  # Titles and axis labels
  plt.title(f"Measuring: {metricName}")
  plt.xlabel("Iterations")
  plt.ylabel(metricName)
  plt.legend(loc="upper right")

  plt.ylim(ylims[0], ylims[1]) 

  #Save the figure
  plt.savefig('bestNew_test_title.png')

  # Show the plot
  plt.show()

  
