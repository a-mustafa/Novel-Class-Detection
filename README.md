# Unsupervised Deep Embedding for Novel Class Detection over Data Stream [1]

**Requires the following python packages:**
  theano, sys, time, io, numpy

**To run the code, type the following command:**
java -cp ECHO_v_0_2_AE.jar:lib/*.jar:. mineClass.Miner -F ~/path/to/Dataset/arfffile.without.arff.extention -S 100 -T 100 -C 80 -N 15 -L 3 -M 25 -B reasc.ReascCtrl -K 25 -P 90 -E "server.py gauss 0.01 2"

**Parameters:**
* -L: Ensumble size
* -S: Initial chunk size (number of instances).
* -C: Classification delay (number of instances).
* -T: Labeling delay (number of instances).
* -E: Autoencoder parameters: Path to the file 'server.py', Noise method, SGD Learning rate, and {1= Update Autoencoder / 2= Do not update Autoencoder }

**The accuracy can be found in output file \*.res:**
* FP%= False Positive Rate
* FN%= False Negative Rate
* Err%= Total Classification Error


[1] Paper Citation TBD
