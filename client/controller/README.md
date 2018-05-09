## OtterTune Controller
The controller is responsible for collecting database metrics and knobs information during an experiment.</br>
#### Usage:
To build the project, run `gradle build`. </br>
To run the controller, you need to provide a configuration file and provide command line arguments (command line arguments are optional). Then run `gradle run`.
 * Configuration file: </br>
   See sample configurations for mysql, postgres and saphana under the `config` directory.
 * Command line arguments:
   * configuration file path (flag : `-c`) </br>
     The path of the input configuration file. You must specify the path of the config file to run the controller.
   * time (flag : `-t`) </br>
     The duration of the experiment in `seconds`. 
     If `t` >= 0, the experiment will last `t` seconds. </br>
     If `t` < 0, the experiment will run forever until it receives `SIGINT`. In the case, after the experiment ends, it will still upload all the result files to the url specified in the configuration file. NOTE: If you want to use `SIGINT` to stop the experiment, please run the experiment by `gradle run <options> --no-daemon` to prevent gradle running on the daemon thread, or it will not receive the `SIGINT` you send. </br>
     The default `t` is set to be -1.
   * output directory (flag : `-d`) </br>
     The path of the output result files. The default path is `./output`.
   * help page (flag : `-h`) </br>
   
 
