/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package reasc;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.Arrays;
import mineClass.Constants;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Ahmad
 */
public class Autoencoder {
    private String cmd;
    private Process p;
    private BufferedReader inp;    
    private BufferedWriter out;
    private Instances deepdata;
    
    public Autoencoder() throws IOException {
        if("".equals(Constants.AEparam)){
            return;
        }        
        this.cmd = "python "+ Constants.AEparam;
        this.p = Runtime.getRuntime().exec(this.cmd);
        this.inp = new BufferedReader( new InputStreamReader(p.getInputStream()) );
        this.out = new BufferedWriter( new OutputStreamWriter(p.getOutputStream()) );        
    }
    
    public String pipe(Instances datapoints, boolean isUpdate, boolean isTrans) {
    String msg="@0";
    if (isUpdate){
        msg="@1";
    }
    if (isTrans){
        msg="@2";
    }
    for(int i=0; i<datapoints.numInstances()-1; i++){
        msg+=Arrays.toString(datapoints.instance(i).toDoubleArray())+"|";
    }
    msg+=Arrays.toString(datapoints.instance(datapoints.numInstances()-1).toDoubleArray())+";";
    String ret;   
    try {
        int indx=0;
        int batchsize=1000;
        while(indx<msg.length()/batchsize){            
            //System.out.println(msg.substring(indx*batchsize, indx*batchsize+batchsize));
            out.write(msg.substring(indx*batchsize, indx*batchsize+batchsize)+ "\n");
            out.flush();
            ret = inp.readLine();
            //System.out.println(ret);
            indx++;
        }
        //System.out.println(msg.substring(indx*batchsize)+ "\n");
        out.write(msg.substring(indx*batchsize)+ "\n");
         //msg + "\n" );
        out.flush();
        ret = inp.readLine().replace("],[","\n");
        //System.out.println("\n" + ret + " 1");
        //while(ret=="[[]]") {
        //  ret = inp.readLine();
        //  System.out.println(ret+"2");
        //}
        //System.out.println(ret+" 1");
        return ret;
    }
    catch (Exception err) {
        err.printStackTrace();
    }
    return "";
    }
    
    public Instances TrainAndTransform(Instances datapoints, boolean isUpdate) {
        if("".equals(Constants.AEparam)){
//            System.out.println("Constants.AEparam in  "+Constants.AEparam);
            return new Instances(datapoints);
        }        
        //Instances deepdata = null;
        if(deepdata!=null){
            deepdata.delete();
        }
        String retdata = pipe(datapoints,isUpdate,false);
        String[] splitedTransData = retdata.split("\\|");
        for(int inst=0; inst<splitedTransData.length; inst++){
          String[] guaranteedOutput=splitedTransData[inst].split("\\,");
          //Double[] doubleValues = Arrays.stream(guaranteedOutput).map(Double::valueOf).toArray(Double[]::new);  
          double[] doubleValues = new double[guaranteedOutput.length];
          for (int dim = 0; dim < doubleValues.length; dim++) {
              doubleValues[dim] = Double.parseDouble(guaranteedOutput[dim]);
          }          
          if(deepdata==null)
          {deepdata=createInstances(doubleValues,datapoints);}
          Instance deepinstance = new Instance(1.0,doubleValues);
          deepdata.add(deepinstance); 
        }       
        System.out.println("AE Updated.. ");
        return deepdata;
    }
    
    public void UpdateAE(Instances datapoints) {
        if("".equals(Constants.AEparam)){
//            System.out.println("Constants.AEparam in  "+Constants.AEparam);
            return;
        }        
        String retdata = pipe(datapoints,true,false);
        System.out.println("AE Updated.. ");
        return;        
    }
    
    public Instances Transform(Instances datapoints) {
        if("".equals(Constants.AEparam)){
//            System.out.println("Constants.AEparam in  "+Constants.AEparam);
            return new Instances(datapoints);
        }
        //Instances deepdata = null;
        if(deepdata!=null){
            deepdata.delete();
        }
        String retdata = pipe(datapoints,false,true);
        String[] splitedTransData = retdata.split("\\|");
        for(int inst=0; inst<splitedTransData.length; inst++){
          String[] guaranteedOutput=splitedTransData[inst].split("\\,");          
          double[] doubleValues = new double[guaranteedOutput.length];
          for (int dim = 0; dim < doubleValues.length; dim++) {
              doubleValues[dim] = Double.parseDouble(guaranteedOutput[dim]);
          }          
//          if(deepdata==null)
//          {deepdata=createInstances(doubleValues,datapoints);}
          Instance deepinstance = new Instance(1.0,doubleValues);
          deepdata.add(deepinstance); 
        }       
        return deepdata;
    }
    
    private Instances createInstances(double[] deepinsdouble, Instances arffinputdata) {
        FastVector deepWekaAttributes = new FastVector();
        for (int att=0; att<deepinsdouble.length-1; att++)
        {
            deepWekaAttributes.addElement(new Attribute("att_"+att));            
        }
        Instances deepdata=new Instances("DeepFeatures", deepWekaAttributes, arffinputdata.numInstances()); /*Ahmad*/
        deepdata.insertAttributeAt(arffinputdata.classAttribute(), deepinsdouble.length-1);
        deepdata.setClassIndex(deepdata.numAttributes()-1);        
        return deepdata;    
    }
    
    public void closeConnections() throws IOException {
        if(this.inp!=null){
            this.inp.close();
            this.out.close();
        }
    }
    
}
