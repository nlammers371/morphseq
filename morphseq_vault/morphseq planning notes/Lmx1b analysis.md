Now that we have done our due diligence and identified well-performing morphVAE architectures, it's high time to use the tools we've been building to do some biology. Revisiting lmx1b crispant analysis provides a nice test case that will allow us to develop conceptual and computational approaches for bridging the gap between sequence and morphology space while also (potentially) answering some interesting biological and technical questions.

## Key questions 

##### Practical considerations (for #Marz)
1. Does he have access to all of the lmx1b experiments? Is Maddy still using both the August and December experiments?
2. Move the excel notebooks to a place where he can access them. Obviously, these are critical for mapping.
3. Consider digging up my old Jupyter notebooks.
4. Make sure the lmx1b images look ok after the most recent pipeline revision

##### Does the morphological "severity" of crispant embryos correlate with the severity of the corresponding transcriptional phenotype?
- Initial thought would be to use distance in PCA space (both morphVAE and sequence space) from WT and/or injection controls.
- Use cell type abundances as our transcriptional readout for now.
- If there *is* a relationship, can we dig into which cell type-level differences are implicated?

##### Disentangling bona-fide changes from differences in developmental timing 

