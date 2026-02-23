```javascript```

---
type: daily
date: Invalid Date
project: #morphSeq
date: Dec-16-2025 2

---

<div style="text-align: center; font-size: 2em; margin-top: 20px;">
    Dec 16 2025 2
</div>
<div style="text-align: center; font-size: 2em; margin-top: 5px;">
    morphSeq
</div>

---
# Key Results
---
# Coding Files Locations
---
# Notes
---

Experiment ids
20251104,20251119,20251121,20251121 have tracked b9d2_pairs (breeding pairs)

while, experiments 20250501,20250519 are b9d2_spawn, as we were not keeping track of pairs at this time, so it represnts a great mixture in the population. 



The B9d2 Phenotype is different thatn the shared phenotype between cep290 and tmem67 phenotypes . Furthermore, as of today i have the data neccesary to analyze this b9d2 phenotypes. 

The questions i will be analyzing are

1. when is ther earliest time i can differentiate any given phenotype-phyologeny (a trend in phenotypes for example always curvy vs curvy then short ) from WT
2. When can i distinguish them from eachother, for example b9d2 phenotypes could have shared phylogeny (indisinguishable for first 24 hours and then when one gets short then they are disinguishable.)


To accopmolish this we are going to use the morphometric analysis measurements over time, curvurture (baseline_deviation_normalized) and total_length_um, to define the population clusters (groups of phenotypically linked embryos) 

- then we are going to see the difference for embeddings, vurvurture, and length, in comparison to wildtrypes, then compare to eachother. 
	- the embedding will be the broadest approch, lookiing at grtoss morphological difference, and the morphometrics will be to analyze when even given difference arrises.
	- we hypothesize that using the emedding we will be able to detect changes that proceed the morphomometrics. 
		- as was done here results/mcolon/20251210_b9d2_earliest_prediction_analysis/README.md
- 
- The utiliies we are going to use is first trajectory analysis and clusterging and difference detection. 
	- we will manually choose k and and which clusters belong to a given phenotype. 
		-  as was done here results/mcolon/20251210_b9d2_earliest_prediction_analysis/README.md
		- using src/analyze/trajectory_analysis/TRAJECTORY_ANALYSIS_README.md
	- Then we are going to see if there is a difference in distribution and classification for these using :
		- src/analyze/difference_detection/distribution
		- src/analyze/difference_detection/classification


for b9d2 there are 2 phenotypes. one is curvurture only, b9d2_pair_5 is the only pair that i have seen with this phenotype 


ill call the curvy which is body axis (BA) then short which is CE (vurvegenet extension) defect, even though  technically my analuysis says they are curvier pre 32hpf. 

specifically there is always a slight kink in the tail and then sometime after 32hpf the CE defect starts to become prevelant . 

### b9d2_pair_4

CE: 
20251121_E03-het
![[file-image-20251216141811941.png]]![[file-image-20251216141823602.png]]
BA:
20251121_G02-homo
![[file-image-20251216142417066.png]]

20251121_D03-homo 
20251121_C05-homo 
20251121_F06-homo
### b9d2_pair_5
BA:
20251121_F04-homo
20251121_G05-homo
20251121_E06-homo

### Non penetrant
b9d2_pair_2 and b9d2_pair_4 have pretty much no penetrance of phenotype (b9d2_pair_2 has just one embryo thats penetrant)


