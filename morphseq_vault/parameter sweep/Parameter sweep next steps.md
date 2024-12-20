#### Initial goals 
The goal of this exercise was to explore hyperparameter space to identify morphVAE model architectures that produce latent spaces with the following features:
1. **Discriminative:** captures biologically relevant differences in embryo morphology (e.g. different stages or perturbations)
2. **Stable:** the overall structure of latent space encodings is robust to the addition or removal of classes/time points from the training set.
3. **Generalizable:** coherently encodes morphologies not included in its training set (related to stability)
4. **Tree-like:** morphologies from different perturbations should start in the same place and branch outward as differences become more apparent.
5. **Robust to technical variability:** this has been less of a focus recently, but it is of critical importance that the latent encodings are as robust as possible to non-biological variability. Foremost among sources of this variability is variation in embryo pose and orientation relative to the imaging plane.


### Summary of results

#Marz has finished a first pass at investigating the performance of a wide range of models. I will only provide a high-level summary here for the sake of brevity. 

**Preliminary conclusions:**
1. **Importance of Gaussian regularization:** the beta parameter, which tunes the strength of the Gaussian regularization term, plays a massive role in the structure and stability of latent space. In particular, a relatively high beta value of 10 (default is 1) seems key to achieving models that are stable (2) and tree-like (4) while still being reasonably (though not optimally) discriminative (1).
2. **Classification-treeness tradeoff:** There is a non-trivial penalty in classification accuracy for the most tree-like/robust models. Likely this mirrors (to some extent) the long-appreciated difference between Autoencoders and Variational Autoencoders: the former has a disjoint latent space that is great for classification but bad for sampling and interpretation. The latter has a much smoother latent space, but this translates to lower classification performance. 
	1. This could be due either to the above-mentioned tradeoff OR to the fact that models with higher beta values may more robust to technical differences between perturbations. If the latter, then this is most definitely a good thing. 
3. **Unavoidable generalization penalty:** all models that generally perform well wrpt the other stated criteria exhibit some level of performance drop-off when applied to novel data. This is fine for now, but bears further investigation.

Overall, #Marz recommends selecting models that optimize for stability and treeness (proxy for interpretability), while still performing well enough across the other metrics of interest. 

#### Metrics and frameworks
As a part of the process of model selection, #Marz developed a collection of metrics and analyses that formalize the qualitative criteria we started out with. In particular, his methods for capturing latent space stability and "co-linearity" (cosine similarity) will be useful down the line. 

Beyond that the principle curve fits, which served as the basis for most of the analyses, will be a great foundation for more biologically-motivated work as we move forward to looking at key datasets of interest.

### Outstanding questions
1. **Test holdout effects etc. specifically on crispants (other than lmx1b)**
	- Start to move toward testing this tool in the context actual use-cases
	- Model might be more robust for encoding less severe morphologies 
	- Consider K-fold test, where a single crispant is held out each time
	
2. **Confirm that chosen model(s) are robust to non-biological variability** 

3. **(low priority) Revisit parameter sweep analysis with spline-independent metrics?**  
	- Spline fits fail for some kinds of latent spaces, and this could lead to biased conclusions regarding regarding model performance
	- Two cases: (i) complex (ball-like, spiral, recursive) latent spaces (ii) general lack of density 
	- **Alternative: merging flux-based metric with local principal curve approach** 