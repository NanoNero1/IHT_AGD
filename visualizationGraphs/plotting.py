import matplotlib.pyplot as plt
import numpy as np
from neptune import Run

plt.figure(figsize=(30,14))


def plotMetric(runID=None,metricName=None,methodNames=None,trials=1):
  
  #Careful with the project variable
  project="dimitri-kachler-workspace/sanity-MNIST"
  api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNWQxNDllOS04OGY1LTRjM2EtYTczZi0xNWI0NTRmZTA1OTEifQ=="
  seeRun = Run(api_token=api_token,project=project, with_id=runID, mode="read-only")
  
  
  # TO-DO: a function for this long line of code? just want the values
  # TO-DO: this is based on the default amount of iterations, depending on the logging it might make a confusing graph
  if metricName != "loss":
    iterations = seeRun[f"trials/0/{methodNames[0]}/iteration"].fetch_values()["value"]
  else:
    iterations = [i for i in range(len( seeRun[f"trials/0/{methodNames[0]}/loss"].fetch_values()["value"] )) ]

  for method in methodNames:
    methodValues = []
    for trial in range(trials):
      trialValues = seeRun[f"trials/{trial}/{method}/{metricName}"].fetch_values()["value"][:len(iterations)]
      methodValues.append(np.array(trialValues))

    #Finding the mean and standard error (see if along one dimension)
    methodMean = np.mean(methodValues,axis=0)
    methodSTD = np.sqrt(np.std(methodValues)) / 10


    plt.plot(iterations[50:], methodMean[50:],label=method)
    #plt.errorbar(iterations, methodMean, yerr = methodSTD)


  # Titles and axis labels
  plt.title(f"Measuring: {metricName}")
  plt.xlabel("Iterations")
  plt.ylabel(metricName)
  plt.legend(loc="upper right")
  plt.show()




  # Show the plot
  plt.show()
