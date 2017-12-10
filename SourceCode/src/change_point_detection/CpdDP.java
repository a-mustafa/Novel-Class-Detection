/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package change_point_detection;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Scanner;
import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.stream.DoubleStream;
import mineClass.Minstances;
import org.apache.commons.math.distribution.BetaDistributionImpl;
import weka.core.Instances;
/**
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Scanner;
import java.io.File;
import org.apache.commons.math.distribution.BetaDistributionImpl;
/**
 *
 * @author axh129430
 */
public class CpdDP {
    public int dim;
    public double secAlgMarSlack;
    public CusumElement[][] cusumElementArr; //to save W(n,k)
    public int trackN[];
    private ArrayList<Double> dynamicWindow;     
    private double gamma; //works as cushion, here i will use gamma >= 30
    private double sensitivity;
    int cushion;
    
    public CpdDP(int dim, double secAlgMarErr, double gamma, double sensitivity){
        this.dim = dim;
        this.secAlgMarSlack = secAlgMarErr;
        this.cusumElementArr = new CusumElement[dim][dim]; 
        trackN = new int[dim];
        this.initArr();
        dynamicWindow = new ArrayList<Double>();
        this.gamma = gamma;
        this.sensitivity = sensitivity;        
    }
    
    public void initArr(){
        for(int i=0; i<this.trackN.length; i++)
        {
            trackN[i] = -1;
        }
    }
    
    public void insertIntoWindow(double value)
    {
        dynamicWindow.add(value);
    }
    
    public ArrayList<Double> getDynamicWindow()
    {
        return this.dynamicWindow;
    }
    
    public void shrinkWindow(int position /* inclusive */)
    {
        for(int index = 0; index <= position; index++)
        {
            dynamicWindow.remove(0);
        }
        /* need to reset this.wVals */
        initArr();
    }
    
    public void shrinkList(ArrayList<Double> list, int position /* inclusive */)
    {
        for(int index = 0; index <= position; index++)
        {
            list.remove(0);
        }
    }
    
    /* functions to estimate beta distribution parameters*/
    //try to supply the sample mean, otherwise calculate
    public double calcBetaDistAlpha(int from, int to, double sampleMean)
    {
        //if sampleMean is not supplied, then calculate sampleMean
        if(sampleMean == -1)
            sampleMean = calculateMean(from, to);
        double sampleVariance = calculateVariance(from, to, sampleMean);
        return ((Math.pow(sampleMean, 2) - Math.pow(sampleMean, 3))/sampleVariance)- sampleMean;
    }
    
    //try to supply the sample mean, otherwise calculate
    public double calculateBetaDistBeta(double alphaPreChange, int from, int to, double sampleMean)
    {
        //if sampleMean is not supplied, then calculate sampleMean
        if(sampleMean == -1)
            sampleMean = calculateMean(from, to);
        return alphaPreChange * ((1/sampleMean)-1);
    }
    /*
     * calculate mean of the elements in dynamicWindow
     * both of the indices from and to are inclusive
     */
    public double calculateMean(int from, int to)
    {
    	double sum = 0.0;
    	for(int i=from; i<=to; i++)
    	{
    		sum += this.dynamicWindow.get(i);
    	}
    	return sum/(to-from+1); 
    }
    
    /*
     * calculate mean of the elements in the list
     * both of the indices from and to are inclusive
     */
    public double calculateListMean(ArrayList<Double> list, int from, int to)
    {
    	double sum = 0.0;
    	for(int i=from; i<=to; i++)
    	{
    		sum += list.get(i);
    	}
    	return sum/(to-from+1); 
    }
    
    public double calculateVariance(int from, int to, double sampleMean)
    {
        double sumOfSquares = 0.0;
        //if sampleMean is not supplied, then calculate sampleMean
        if(sampleMean == -1)
            sampleMean = calculateMean(from, to);
        for(int i=from; i<=to; i++)
        {
            sumOfSquares += (this.dynamicWindow.get(i) - sampleMean) * (this.dynamicWindow.get(i) - sampleMean);
        }
        return sumOfSquares/(to-from+1); 
    }
    
    public int/*estimated change point*/ detectChange() throws Exception{
        int estimatedChangePoint = -1;
        int N = this.dynamicWindow.size();
        this.cushion = Math.max(100, (int)Math.floor(Math.pow(N, gamma)));
        //mean conf. should not fall below 0.3
        double preChangeMean, postChangeMean, wholeMean;
        wholeMean = calculateMean(0, N-1);
        if((N>(2*this.cushion) && wholeMean <= 0.3) || this.dynamicWindow.size() > this.dim)
            return N-1;
        double threshold = -Math.log(this.sensitivity);
        double w = 0;
        int kAtMaxW = -1;
        for(int k = this.cushion; k<= N-this.cushion; k++){
            double skn = 0;
            int prevN = this.trackN[k];
            preChangeMean = prevN==-1?calculateMean(0, k-1):this.cusumElementArr[prevN][k].getPreChangeMean();
            postChangeMean = calculateMean(k, N-1);
            if(postChangeMean <= (1-this.secAlgMarSlack)*preChangeMean){//signal from secondary
                if(prevN == -1){
                    //calculate from scratch
                    /* estimate pre and post change parameters */
                    double alphaPreChange = calcBetaDistAlpha(0, k-1, preChangeMean);
                    double betaPreChange = calculateBetaDistBeta(alphaPreChange, 0, k-1, preChangeMean);
                    double alphaPostChange = calcBetaDistAlpha(k, N-1, postChangeMean);
                    double betaPostChange = calculateBetaDistBeta(alphaPostChange, k, N-1, postChangeMean);
                    BetaDistributionImpl preBetaDist = new BetaDistributionImpl(alphaPreChange, betaPreChange);
                    BetaDistributionImpl postBetaDist = new BetaDistributionImpl(alphaPostChange, betaPostChange);
                    for(int i=k; i<N; i++)
                    {
                        try{
                            skn += Math.log(postBetaDist.density(this.dynamicWindow.get(i).doubleValue())/preBetaDist.density(this.dynamicWindow.get(i).doubleValue()));
                        }
                        catch(Exception e){
                            e.printStackTrace();
                            System.out.println("continuing...");
                            skn = 0;
                            break;
                        }
                    }
                    this.cusumElementArr[N-1][k] = new CusumElement(preBetaDist, postBetaDist, preChangeMean, skn);  
                }
                else{//warning and calculate recursively
                    double alphaPostChange2 = calcBetaDistAlpha(k, N-1, postChangeMean);
                    double betaPostChange2 = calculateBetaDistBeta(alphaPostChange2, k, N-1, postChangeMean);
                    BetaDistributionImpl postBetaDist2 = new BetaDistributionImpl(alphaPostChange2, betaPostChange2);
                    skn += this.cusumElementArr[prevN][k].getCusumScore();
                    for(int i=prevN+1; i<N; i++)
                    {
                        try{
                            skn += Math.log(postBetaDist2.density(this.dynamicWindow.get(i).doubleValue())/this.cusumElementArr[prevN][k].getPreChangeDist().density(this.dynamicWindow.get(i).doubleValue()));
                        }
                        catch(Exception e){
                            e.printStackTrace();
                            System.out.println("continuing...");
                            skn = 0;
                            break;
                        }
                    }
                    this.cusumElementArr[N-1][k] = new CusumElement(this.cusumElementArr[prevN][k].getPreChangeDist(), postBetaDist2, preChangeMean, skn); 
                }
                this.trackN[k] = N-1;
            }
            if(skn > w)
            {
                w = skn;
                kAtMaxW = k;
            }    
        }
        if(w >= threshold && kAtMaxW != -1)
        {
            System.out.println("\nChangePoint Found!");
            estimatedChangePoint = kAtMaxW;
            System.out.println("Estimated change point is " + estimatedChangePoint+", detected at point: "+ N);
        }
        //force change point if confidence falls down terribly
        if(estimatedChangePoint == -1 && N>=100 && wholeMean < 0.3)
            estimatedChangePoint = N-1;
        return estimatedChangePoint;
    }
    
    public int/*estimated change point*/ detectChangenonpar() throws Exception
    {
        int estimatedChangePoint = -1;
        int N = this.dynamicWindow.size(); //     n = sample size
        this.cushion = Math.max(100, (int)Math.floor(Math.pow(N, gamma)));
        int r = 4;                       //     r = number of finite partition intervals
        double max = Collections.max(this.dynamicWindow);
        double min = Collections.min(this.dynamicWindow);
        double d = (max-min)/r;         //     d = length of partition intervals                   
        double b = min + d;             //    -a = the 1st partition point
        double c = max - d;             //     a = the last partition point
        double eps = 0.7;        
        //Compute Hist
        double[][] xi = new double[N][r]; // xi contains the histogram

        for(int i=0; i<N; i++){
            if(this.dynamicWindow.get(i) <= b){
                    xi[i][0] = 1;
            }
            else{
                if(this.dynamicWindow.get(i) > c ){
                        xi[i][r-1]=1;
                }
                else{
                    for(int m=1; m<r; m++){
                        if(this.dynamicWindow.get(i) > b+d*(m-1) && this.dynamicWindow.get(i) <= b+d*(m)){
                              xi[i][m] = 1;
                              break;
                        }
                    }
                }
            }
        }
        
//	Compute S_k        
        double w = 0;
        double Skn = 0;
        int kAtMaxW = -1;
        double threshold = -Math.log(this.sensitivity);
        for(int k = this.cushion; k<= N-this.cushion; k++){            
            double[] p = getMeanValue(Arrays.copyOfRange(xi,0,k));	    	//mean(xi(0:k,:));
            double[] q = getMeanValue(Arrays.copyOfRange(xi,k+1,N));		//mean(xi(k+1:n,:));
            double s = 0;
            for(int i=0; i<p.length; i++){
                    if(p[i]==0.0){
                            s++;	// number of times p = 0 %
                    }
            }
            double ep= eps/(N-k);
            double zer = 0.0001;

            double[] p1 = new double[p.length];
            for(int i=0; i<p1.length; i++){
                if(p[i]<zer){
                        p1[i]= 1 + p[i];
                }else{
                        p1[i]= p[i];
                }
            }
            double[] q1 = new double[q.length];
            for(int i=0; i<q1.length; i++){
                if(q[i]<zer){
                        q1[i]= 1 + q[i];
                }else{
                        q1[i]= q[i];
                }
            }
            double s1 = 0;
            if(s == 0){
                s1 = 1 + s;
            }else{
                s1 = s;
            }
//            z = (q>zer).*q1.*((s==0)*log(q1./p1) + (s>0)*((p<zer).*log(q1/ep*s1) + (p>zer).*log(q1./p1/(1-ep))));
            try
            {
                double[] z = new double[p.length];
                if(s>0){
                    for(int i=0; i<z.length; i++){
                        if(p[i]<zer){
                            z[i]= Math.log(q1[i]/ep*s1);
                        }else{
                            z[i]= Math.log(q1[i]/p1[i]/(1-ep));
                        }
                    }
                }else{
                    if(s==0){
                        for(int i=0; i<z.length; i++){
                            z[i] = Math.log(q1[i]/p1[i]);
                        }
                    }
                }
                for(int i=0; i<q.length; i++){
                    if(q[i]>zer){
                            z[i] = q1[i] * z[i]; 
                    }else{
                            z[i] =0;
                    }
                }            
//              Skn = k*getSumValue(z);            
                Skn = k* Arrays.stream(z).sum();
            }
            catch(Exception e){
                e.printStackTrace();
                System.out.println("continuing...");
                Skn = 0;
                break;
            }
//          w = Math.max(w,Skn);
            if(Skn > w)
            {
                w = Skn;
                kAtMaxW = k;
            }
	}
        if(w >= threshold && kAtMaxW != -1)
        {
            System.out.println("\nChangePoint Found!");
            estimatedChangePoint = kAtMaxW;
            System.out.println("Estimated change point is " + estimatedChangePoint);
        }
        return estimatedChangePoint;
    }
    
    // getting the mean value
    private double[] getMeanValue(double[][] array){
            double[] means = new double[array[0].length];
            for(int dim=0; dim<array[0].length; dim++){
                  double SUM = array[0][dim];
                  for(int i=1;i < array.length;i++){
                      SUM+=array[i][dim];
                 }
                  means[dim]=SUM/array.length;
            }
     return means;
    }
    
    
    public int/*estimated change point*/ detectChangeNonparMultidim(Instances dataPoints) throws Exception
    {
        int estimatedChangePoint = -1;
        int N = dataPoints.numInstances(); //     n = sample size
        this.cushion = Math.max(100, (int)Math.floor(Math.pow(N, gamma)));
        int r = 4;                       //     r = number of finite partition intervals
        double eps = 0.7; 
        double w = 0;
        double Skn = 0;
        int kAtMaxW = -1;
        double threshold = -Math.log(this.sensitivity);
        
        for(int att=0; att<dataPoints.numAttributes(); att++){
            double[] dataPoints_attd=dataPoints.attributeToDoubleArray(att);
            ArrayList<Double> dataPoints_att=new ArrayList<>();
            for(int ins=0;ins<dataPoints_attd.length;ins++){
                dataPoints_att.add(dataPoints_attd[ins]);
            }
            double max = Collections.max(dataPoints_att);
            double min = Collections.min(dataPoints_att);
            double d = (max-min)/r;         //     d = length of partition intervals                   
            double b = min + d;             //    -a = the 1st partition point
            double c = max - d;             //     a = the last partition point

            //Compute Hist
            double[][] xi = new double[N][r]; // xi contains the histogram

            for(int i=0; i<N; i++){
                if(dataPoints_att.get(i) <= b){
                        xi[i][0] = 1;
                }
                else{
                    if(dataPoints_att.get(i) > c ){
                            xi[i][r-1]=1;
                    }
                    else{
                        for(int m=1; m<r; m++){
                            if(dataPoints_att.get(i) > b+d*(m-1) && dataPoints_att.get(i) <= b+d*(m)){
                                  xi[i][m] = 1;
                                  break;
                            }
                        }
                    }
                }
            }

            //	Compute S_k  
            for(int k = this.cushion; k<= N-this.cushion; k++){            
                double[] p = getMeanValue(Arrays.copyOfRange(xi,0,k));	    	//mean(xi(0:k,:));
                double[] q = getMeanValue(Arrays.copyOfRange(xi,k+1,N));		//mean(xi(k+1:n,:));
                double s = 0;
                for(int i=0; i<p.length; i++){
                        if(p[i]==0.0){
                                s++;	// number of times p = 0 %
                        }
                }
                double ep= eps/(N-k);
                double zer = 0.0001;

                double[] p1 = new double[p.length];
                for(int i=0; i<p1.length; i++){
                    if(p[i]<zer){
                            p1[i]= 1 + p[i];
                    }else{
                            p1[i]= p[i];
                    }
                }
                double[] q1 = new double[q.length];
                for(int i=0; i<q1.length; i++){
                    if(q[i]<zer){
                            q1[i]= 1 + q[i];
                    }else{
                            q1[i]= q[i];
                    }
                }
                double s1 = 0;
                if(s == 0){
                    s1 = 1 + s;
                }else{
                    s1 = s;
                }
    //            z = (q>zer).*q1.*((s==0)*log(q1./p1) + (s>0)*((p<zer).*log(q1/ep*s1) + (p>zer).*log(q1./p1/(1-ep))));
                try
                {
                    double[] z = new double[p.length];
                    if(s>0){
                        for(int i=0; i<z.length; i++){
                            if(p[i]<zer){
                                z[i]= Math.log(q1[i]/ep*s1);
                            }else{
                                z[i]= Math.log(q1[i]/p1[i]/(1-ep));
                            }
                        }
                    }else{
                        if(s==0){
                            for(int i=0; i<z.length; i++){
                                z[i] = Math.log(q1[i]/p1[i]);
                            }
                        }
                    }
                    for(int i=0; i<q.length; i++){
                        if(q[i]>zer){
                                z[i] = q1[i] * z[i]; 
                        }else{
                                z[i] =0;
                        }
                    }            
    //              Skn = k*getSumValue(z);            
                    Skn = k* Arrays.stream(z).sum();
                }
                catch(Exception e){
                    e.printStackTrace();
                    System.out.println("continuing...");
                    Skn = 0;
                    break;
                }
    //          w = Math.max(w,Skn);
                if(Skn > w)
                {
                    w = Skn;
                    kAtMaxW = k;
                }
            }
            if(w >= threshold && kAtMaxW != -1)
            {
                System.out.println("\nChangePoint Found!");
                estimatedChangePoint = kAtMaxW;
                System.out.println("Estimated change point is " + estimatedChangePoint);
            }
        }
        
        return estimatedChangePoint;
    }
    
    public int/*estimated change point*/ detectChangeNonparMultidim(Minstances dataPoints) throws Exception
    {
        int estimatedChangePoint = -1;
        int N = dataPoints.numInstances(); //     n = sample size
        this.cushion = Math.max(100, (int)Math.floor(Math.pow(N, gamma)));
        int r = 4;                       //     r = number of finite partition intervals
        double eps = 0.7; 
        double w = 0;
        double Skn = 0;
        int kAtMaxW = -1;
        double threshold = -Math.log(this.sensitivity);
        
        for(int att=0; att<dataPoints.numAttributes(); att++){
            double[] dataPoints_attd=dataPoints.attributeToDoubleArray(att);
            ArrayList<Double> dataPoints_att=new ArrayList<>();
            for(int ins=0;ins<dataPoints_attd.length;ins++){
                dataPoints_att.add(dataPoints_attd[ins]);
            }
            double max = Collections.max(dataPoints_att);
            double min = Collections.min(dataPoints_att);
            double d = (max-min)/r;         //     d = length of partition intervals                   
            double b = min + d;             //    -a = the 1st partition point
            double c = max - d;             //     a = the last partition point

            //Compute Hist
            double[][] xi = new double[N][r]; // xi contains the histogram

            for(int i=0; i<N; i++){
                if(dataPoints_att.get(i) <= b){
                        xi[i][0] = 1;
                }
                else{
                    if(dataPoints_att.get(i) > c ){
                            xi[i][r-1]=1;
                    }
                    else{
                        for(int m=1; m<r; m++){
                            if(dataPoints_att.get(i) > b+d*(m-1) && dataPoints_att.get(i) <= b+d*(m)){
                                  xi[i][m] = 1;
                                  break;
                            }
                        }
                    }
                }
            }

            //	Compute S_k  
            for(int k = this.cushion; k<= N-this.cushion; k++){            
                double[] p = getMeanValue(Arrays.copyOfRange(xi,0,k));	    	//mean(xi(0:k,:));
                double[] q = getMeanValue(Arrays.copyOfRange(xi,k+1,N));		//mean(xi(k+1:n,:));
                double s = 0;
                for(int i=0; i<p.length; i++){
                        if(p[i]==0.0){
                                s++;	// number of times p = 0 %
                        }
                }
                double ep= eps/(N-k);
                double zer = 0.0001;

                double[] p1 = new double[p.length];
                for(int i=0; i<p1.length; i++){
                    if(p[i]<zer){
                            p1[i]= 1 + p[i];
                    }else{
                            p1[i]= p[i];
                    }
                }
                double[] q1 = new double[q.length];
                for(int i=0; i<q1.length; i++){
                    if(q[i]<zer){
                            q1[i]= 1 + q[i];
                    }else{
                            q1[i]= q[i];
                    }
                }
                double s1 = 0;
                if(s == 0){
                    s1 = 1 + s;
                }else{
                    s1 = s;
                }
    //            z = (q>zer).*q1.*((s==0)*log(q1./p1) + (s>0)*((p<zer).*log(q1/ep*s1) + (p>zer).*log(q1./p1/(1-ep))));
                try
                {
                    double[] z = new double[p.length];
                    if(s>0){
                        for(int i=0; i<z.length; i++){
                            if(p[i]<zer){
                                z[i]= Math.log(q1[i]/ep*s1);
                            }else{
                                z[i]= Math.log(q1[i]/p1[i]/(1-ep));
                            }
                        }
                    }else{
                        if(s==0){
                            for(int i=0; i<z.length; i++){
                                z[i] = Math.log(q1[i]/p1[i]);
                            }
                        }
                    }
                    for(int i=0; i<q.length; i++){
                        if(q[i]>zer){
                                z[i] = q1[i] * z[i]; 
                        }else{
                                z[i] =0;
                        }
                    }            
    //              Skn = k*getSumValue(z);            
                    Skn = k* Arrays.stream(z).sum();
                }
                catch(Exception e){
                    e.printStackTrace();
                    System.out.println("continuing...");
                    Skn = 0;
                    break;
                }
    //          w = Math.max(w,Skn);
                if(Skn > w)
                {
                    w = Skn;
                    kAtMaxW = k;
                }
            }
            if(w >= threshold && kAtMaxW != -1)
            {
                System.out.println("\nChangePoint Found!");
                estimatedChangePoint = kAtMaxW;
                System.out.println("Estimated change point is " + estimatedChangePoint);
            }
        }
        
        return estimatedChangePoint;
    }
}
