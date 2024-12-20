```javascript```

---
type: daily
date: 2024-12-17
project: #morphSeq
date: Dec-17-2024

---

<div style="text-align: center; font-size: 2em; margin-top: 20px;">
    Dec 17 2024
</div>
<div style="text-align: center; font-size: 2em; margin-top: 5px;">
    MorphSeq Param Sweep with Splines
</div>

---
# Previous Paramater Sweep (where we left off)
This is following up on the previous paramater sweep. Since then I have generated splines and different ways to measure colinearity WITHIN and across the splines for perturbations.
**Main takeaways from previous param sweep:** 
![[Projects/morphSeq/Journal/Nov-2024/MorphSeq Paramater Sweep Takeaways;Nov-28-2024#Main Takeawats]]
---

# Main Takeaway
##  Key Players
I used splines using [[@einbeckLocalPrincipalCurves20E05]] to fit splines to the data we wanted to Measure 

- Tree Likeness (Prettiness), done by looking at dispersion (distance to centroid) of splines
	- dispersion_first_n and dispersion_last_n is how close the points are in the first n points along the datasets splines (or lastn)
		- Note i used n=5
	- disp_coefficient  is a fitting a linear model to the dispersion ALONG the spline
		- This is to make sure that splines ( and therefore points) are spreading out over timne
	- SegmentColiniartity_mean_within (all or hld): point moving in same direction
		- This is looking WITHIN a dataset and looking at pairwise between perts how these points move in the same direction (cosine similiarty)
- Robustness
	- SegColinearity, is looking at a model INSTANCE (each model has all or hld (holding out lmx1b and gdf3)) and then looking at how for the same pert they move in the same direction. 
	- Aligned MSE using the quaternion implementaion of Kabsch aligning algorithm we look at how the MSE between like perturbation for a model instance is.
		- This is very stringent but a good measure. 
	- Initial MSE
		- How without aligning the MSE
		- used as an internal check to make sure Alignmen tis working AND to see if alignment is actually necessary. 
- Classification:
	- F1 Score a measure of how good model is at classification
		- for more info see previous param sweep (referenced above)
## Conclusion 

### Chosen Model (74):
I think Model 74 (or another model in its class should be choosen for these reasons)
![[MorphSeq Param Sweep with Splines;Dec-17-2024#Summary of why model 74]]

### Tradeoff of Prettiness vs Classificaion
- There is a tradeoff on prettiness vs classification accuracy, though this is slight.
It is MOST definitely worth the small decrease in classification accuracy to get a more intepretable morphology space. 
	- See [[MorphSeq Param Sweep with Splines;Dec-17-2024#Tradeoff of Prettiness vs Classificaion]] for more detail 

### Beta Drives Prettiness AND Robustness
-  It can be seen that the Beta variable drives prettiness and robustness
- Higher values of the beta variable makes  the model make the individual dimensions in the embedding space more iid (independent)
	- It also clearly has the property of making morphology space prettier
		- potentially by smoothing things out. 
![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 77.png|300]]
- This is a surprising result as prettiness/tree likeness is NOT garunteed to be correlated with robustnes/stability.
- High beta model are so rubst that they actually dont need to be aligned! (tgough you should still align them lol) 
### New Spline Measurement do capture things we wanted them too
- Implementing splines allowed us to finally get at what was the drivers of prettiness
	- previously not able to 
### Concerns
- sometimes splines dont work, but this is when they arent pretty (tight turns, not tree shaped) see model 98
- but these might be inflating the  ALigned MSE values but not really worried about this. 
	- because when they fail its because data istn tree like. 
## Next Steps
- Consider other measurement that dont need splines  (none come to mind)
- I see no obvious change, but if were to make the dispersion_metric first_n and lastn could be calculated from one linear equation, the y intercept oculd be first_n and the multiple the slope by the last index (wich is the dis cofeffiicent here actually) and then  add the slope to get last _n
- Of course we need to find way of projecting points onto the spline and then time across it
	- measure variance accross spline
	- measure variablity of time across spline



---
# Collinearity
- Here we see that if a model is not colinear within a dataset it wont be in the other,
- Furthermore, that if it has a high colinearity within one then it will be colinear with the held out data! (this can be seen by colorying the points by ACROSS dataset colinearity). This is the first actual evaluation of 
- this means that if you project into a space that is alrerady linear, you will get linear manfinolds.
	- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 7.png]]
### beta as causative for collinearity AND Robustness
![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 14.png|300]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 19.png|300]]






![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 13.png|350]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 20.png|350]]
![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 21.png|350]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 22.png|350]]
- note that NT-Xent has tighter distribtuions around y=x (more stable)

-----
- lmx1b
![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 17.png|300]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 25.png|300]]
- gdf3
![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 18.png|300]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 26.png|300]]


![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 15.png|300][[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 16.png|300]]


![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 24.png|300]]

### MSE Initial -> Aligned
- Here I Noticed that for one of the models that did the best in terms of colinearity metrics, that the PCA was RIGHT on top of eachother pre alignemnt which was not garunteed at all. (this would imply that PCA is generating the same eignevectors acorss training regimes)
- example 71 is a prime example! 
	- [[Projects/morphSeq/Journal/Nov-2024/MorphSeq Paramater Sweep Takeaways;Nov-28-2024#Classification|MorphSeq Paramater Sweep Takeaways;Nov-28-2024]]
	- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 4.png]]
- This is confirmed here! that yes if the model is colinear then it will already by aligned!
	- Note that all the lines are under the y=x line which means Initial RMSE < Aligned RMSE 
	- (second plot is removeing outliers )
	- 
- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 5.png|400]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 6.png|400]]

# Dispersion (tree likeness)
- we wanted to see how tree like our models by looking at the first n  and last n points and seeing hoe spread apart they are, we want firs n to be close to zero and last n to be bigger. Additionally we use dispersion coefficient which simply looks at dispersion along the spline. higher dispersion_coef measn higher speard across the spline.  (note n = 5)
- main takeway from these plot is that as long as the model is average or better at the segment collinearity metrics then it will will be tree and that generally dispersion increases over the spline. 

- here we see that for models that do ok with linearity metrics  have a first_n close to 0 AND last n around 2
	- also note that when held out model the beginning points (first_n) become more spread out as well 
-  ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 11.png|300]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 12.png|300]]
-  ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 10.png|500]]
- We can see that model becomes slightly more disperse (higher dispersion coefficient in hld) when holding out perturbations. 
	- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 9.png|500]]
----
#### drivers of Dispersion: 
- Higher beta --> more linear --> closer first_n --> more tree like
![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 44.png|300]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 46.png|300]]



# Prettiness vs Classification Accuracy 
 - clear trend that:
	1. classification can come at the cost of prettiness and vice versa
	2. for a given f1 score more colinearirity decreases classication robustness  

- coloring segcoliniarity by F1 score showes that there is generally a tradeoff that lower F1 scores (bluer) corresponds to higher colinearity metrics
	- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 27.png|500]]
- Hard to draw a trend between F1 score difference (robustness) and  seg colinearity 
	- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 28.png|500]]
- Slight signal but you can SEE that for a ==given F1 score== higher colinearity means that there wil be more of a perforance loss wrt. F1 score robustness... 
	- though the tradeodd isnt anything crazy...
	- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 29.png|500]]

## Classification on per Pert level and Prettiness
- Trends are recapitulated, tradeoff between prettiness and classification
	![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 112.png|300]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 113.png|300]]



# Picking models

##  First pass with metrics we care about F1 and Align MSE (prettiness)

 - I think having **F1 score > .7 and Aligned MSE < 1** are good acceptable models (look at blue and red lines)
- We care about F1 score (classification), Aligned RMSE (robustness AND pretiness), I am coloring by other things too, to make sure they have good properties.
- Note we know Aligned RMSE is associated with robustness AND pretiness) because of [[MorphSeq Param Sweep with Splines;Dec-17-2024#MSE Initial -> Aligned]]
![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 72.png]]
 ### Sel Models from different parts of the plot used to select models: 
		using Aligned RMSE and F1 score as discriminators
-  Best models (98,48,19,74,77)
	- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 66.png]]
- Second Tranche (97,98)
	- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 67.png]]
-  Third Tranche (11,43,52)
	- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 70.png]]
- Fourth Tranche (67, 184, 108)
	- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 71.png]]

---
	
- assuring models have nice segment colinearity
	- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 73.png|400]]
- assuring models have tree like structure low disp_first_n and non abnormal dispersion_coefficient 
	-  ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 75.png|300]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 76.png|300]]
	-  ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 78.png|300]]
- we can see beta is driving the prettiness
	- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 77.png|300]]
---
Extracting the top model models (98,48,19,74,77):
- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 66.png]]
	- None of them use the time only flag, and ALL of them are NT-Xent
- we can see that model is in top quartile for Classification metrics and has very close first_n, and is average in terms of segment colinearity
	- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 41.png|200]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 42.png|200]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 43.png|200]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 47.png|200]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 48.png|200]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 49.png|200]]
### Looking at Best Models (Lower right of plot)



#### Sel Model_Index 98:
- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 79.png|300]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 80.png|300]]
- ![[f1_score_over_time_multiclass_model__98_all_perts_F1.png|300]]![[f1_score_over_time_multiclass_model__98_hld_gdf3_lmx1b_perts_F1 1.png|300]]
#### Sel Model_Index 48  
- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 83.png|300]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 82.png|300]]
- ![[f1_score_over_time_multiclass_model__48_all_perts_F1.png|300]]![[f1_score_over_time_multiclass_model__48_hld_gdf3_lmx1b_perts_F1.png|300]]
#### Sel Model_Index 19
- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 86.png|300]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 85.png|300]]
- ![[f1_score_over_time_multiclass_model__19_all_perts_F1.png|300]]![[f1_score_over_time_multiclass_model__19_hld_gdf3_lmx1b_perts_F1.png|300]]
#### Sel Model_Index 74 
- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 87.png|250]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 88.png|250]]
- ![[f1_score_over_time_multiclass_model__74_all_perts_F1.png|250]]![[f1_score_over_time_multiclass_model__74_hld_gdf3_lmx1b_perts_F1.png|250]]

#### Sel Model_Index  77 
- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 89.png|300]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 91.png|300]]
- ![[f1_score_over_time_multiclass_model__77_all_perts_F1.png|300]]![[f1_score_over_time_multiclass_model__77_hld_gdf3_lmx1b_perts_F1.png|300]]

### Looking at Second Traunche (97,98; Top  Right) 
- These models have the highest F1 score 97 has .78 which is the highers
#### Sel Model_Index 97 
- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 92.png|300]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 93.png|300]]
- ![[f1_score_over_time_multiclass_model__97_all_perts_F1.png|300]]![[f1_score_over_time_multiclass_model__97_hld_gdf3_lmx1b_perts_F1.png|300]]
#### Sel Model_Index 98 
- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 94.png|300]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 95.png|300]]
-  ![[f1_score_over_time_multiclass_model__98_all_perts_F1 1.png|300]]![[f1_score_over_time_multiclass_model__98_hld_gdf3_lmx1b_perts_F1 2.png|300]]

### Looking at Third Tranche  (11,43,52; Lower left)
#### Sel Model_Index11 
- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 100.png|300]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 101.png|300]]
- ![[f1_score_over_time_multiclass_model__11_all_perts_F1 1.png|300]]![[f1_score_over_time_multiclass_model__11_hld_gdf3_lmx1b_perts_F1.png|300]]
#### Sel Model_Index 43 
- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 102.png|300]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 103.png|300]]
- ![[f1_score_over_time_multiclass_model__43_all_perts_F1.png|300]] ![[f1_score_over_time_multiclass_model__43_hld_gdf3_lmx1b_perts_F1.png|300]]
#### Sel Model_Index 52
- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 104.png|300]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 105.png|300]]
- ![[f1_score_over_time_multiclass_model__52_all_perts_F1.png|300]]![[f1_score_over_time_multiclass_model__52_hld_gdf3_lmx1b_perts_F1.png|300]]
### Looking at  Fourth Tranche (67, 184, 108; Top Left)
#### Sel Model_Index 67
- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 106.png|300]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 107.png|300]]
- ![[f1_score_over_time_multiclass_model__67_all_perts_F1.png|300]]![[f1_score_over_time_multiclass_model__67_hld_gdf3_lmx1b_perts_F1.png|300]]
#### Sel Model_Index 184
- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 108.png|300]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 109.png|300]]
- ![[f1_score_over_time_multiclass_model__184_all_perts_F1 1.png|300]]![[f1_score_over_time_multiclass_model__184_hld_gdf3_lmx1b_perts_F1.png|300]]
#### Sel Model_Index 108
- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 111.png|300]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 110.png|300]]
- ![[f1_score_over_time_multiclass_model__108_all_perts_F1.png|300]] ![[f1_score_over_time_multiclass_model__108_hld_gdf3_lmx1b_perts_F1.png|300]]
---
## Choosing Model 74
### Summary of why model 74 
- I choose model 74 because it still has tree like structure, is stable, and it has good classification accuracy 
- Reading below we see that it does a better job extracting data than other models e.g. model 97 (not neccessarily those in its same tanche e.g. 77) 
- It even does better than the previous model we were using!!! 
![[MorphSeq Param Sweep with Splines;Dec-17-2024#Sel Model_Index 74]]


### comparing it to models with higher classification [[MorphSeq Param Sweep with Splines;Dec-17-2024#Looking at Second Traunche (97,98; Top Right)]]

Pretinness:
- Obviously these models with classifcation just have more compact less  and less tree like morphology space (notice how the perts dont branch out)
	- Model 74 wins this easily
Now Classification is more nuanced but it still LOSES:
- The model 97 (and 98) seem to really be edging out the improved F1 scores through improvements in lmx1b.
- However theres two things that should be noted:
1. That the gains in lmx1b in earlier time points arent learned when held out
2. In all  Perts other than Lmx1b the model 97 is doing WORSE
- These two observations make me suspicious that it is learning non-biological information. If the model couldve repeated this performance in the hld I wouldve beleived it was possibly learning something useful. 

Model 97
- ![[MorphSeq Param Sweep with Splines;Dec-17-2024#Sel Model_Index 97]]
Model 74
- ![[f1_score_over_time_multiclass_model__74_all_perts_F1.png|300]]![[f1_score_over_time_multiclass_model__74_hld_gdf3_lmx1b_perts_F1.png|300]]

### comparing 74 to previous model we were using 
- First striking takeway is that model 74 does a much better job of EXCTRACTING relevant information for classification at earlier time points in comparison to previous model. (note that all the best models do a better job not just 74)
	- compare the purple lines (gdf3)
- Prev model:
	- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 99.png|300]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 98.png|300]]
- Model 74:
	-  ![[f1_score_over_time_multiclass_model__77_all_perts_F1.png|300]]![[f1_score_over_time_multiclass_model__77_hld_gdf3_lmx1b_perts_F1.png|300]]

### Following up on gains on Classification
- I was curious if increasing the tol  (tol: 0.05 --> .10e_4) of classsifcation (letting log reg model run longer) will improve results and the answer is not really by much
	- Increasing the tol of classification:
	- ![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 97.png|300]]![[Projects/morphSeq/Journal/Dec-2024/attachments/MorphSeq Param Sweep with Splines;Dec-17-2024/image 96.png|300]]	



