/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package mineClass;

//Save results per 1K instance
public class ResultStat{
    public int fp = 0;     //false positive
    public int fn = 0;     //false negative
    public int nc = 0;     //novel class
    public int err = 0;    //error
    public int total = 0;  //total instances processed
    public boolean printed = false;
    
    public int tfp = 0;     //false positive
    public int tfn = 0;     //false negative
    public int zeroAttack=0;
    
    //Regular classification metrics /*Ahmad*/
    public int cfp = 0;     //false positive 
    public int cfn = 0;     //false negative
    public int ctp = 0;     //true positive
    public int ctn = 0;     //true negative
    
    public void addStat(Minstance inst)
    {
        if(inst.tfp)
            tfp ++;
        else if(inst.tfn)
                tfn ++;
        if(inst.cfp)
            cfp ++;
        else if(inst.cfn)
                cfn ++;
        else if(inst.ctp)
                ctp ++;
        else if(inst.ctn)
                ctn ++;
        if(inst.fp)
            fp ++;
        else if(inst.fn)
                fn ++;
        if(inst.err)
            err ++;
        if(inst.isNovel)
            nc ++;
        if(inst.isZeroAttack)
            zeroAttack++;
        inst.Comitted = true;
        //String debug = inst.Id + " "+inst.fp+" "+inst.fn+" "+inst.isNovel+" "+inst.err + " "+(int)inst.EPrediction.Predclass + " "+ (int)inst.classValue()+ " "+inst.Predictions.size();
        //Constants.logger.debug(debug);
        //debug = "";
        for(int i = 0; i < inst.Predictions.size(); i ++)
        {
            MapPrediction op = (MapPrediction)inst.Predictions.get(i);
            //Constants.logger.debug(op.Cid+" "+op.Predclass+ " "+op.Isoutlier +" "+Math.exp(-op.Dist));
        }
        total ++;
    }

    public boolean full()
    {
        return total == 1000;
    }
}
