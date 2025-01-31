library(monocle3)
library(plotly)
library(dplyr)

comb<-load_monocle_objects("/net/trapnell/vol1/home/waltno/REF1/REF1_scale_coembed")
comb$dataset[comb$dataset== 2]<-"SCALE"
comb$dataset[comb$expt== "REF1"]<-"REF1"
comb$dataset[comb$dataset== 1]<-"REF"