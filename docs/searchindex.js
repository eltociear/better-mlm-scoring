Search.setIndex({docnames:["index","minicons","minicons.cwe","minicons.scorer","minicons.supervised","minicons.utils","modules","representations","surprisals"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["index.rst","minicons.rst","minicons.cwe.rst","minicons.scorer.rst","minicons.supervised.rst","minicons.utils.rst","modules.rst","representations.md","surprisals.md"],objects:{"":[[1,0,0,"-","minicons"]],"minicons.cwe":[[2,1,1,"","CWE"]],"minicons.cwe.CWE":[[2,2,1,"","encode_text"],[2,2,1,"","extract_paired_representations"],[2,2,1,"","extract_representation"],[2,2,1,"","extract_sentence_representation"]],"minicons.scorer":[[3,1,1,"","IncrementalLMScorer"],[3,1,1,"","LMScorer"],[3,1,1,"","MaskedLMScorer"]],"minicons.scorer.IncrementalLMScorer":[[3,2,1,"","add_special_tokens"],[3,2,1,"","compute_stats"],[3,2,1,"","distribution"],[3,2,1,"","encode"],[3,2,1,"","logprobs"],[3,2,1,"","next_word_distribution"],[3,2,1,"","prepare_text"],[3,2,1,"","prime_text"],[3,2,1,"","sequence_score"],[3,2,1,"","token_score"]],"minicons.scorer.LMScorer":[[3,2,1,"","adapt_score"],[3,2,1,"","add_special_tokens"],[3,2,1,"","compute_stats"],[3,2,1,"","decode"],[3,2,1,"","distribution"],[3,2,1,"","encode"],[3,2,1,"","logprobs"],[3,2,1,"","partial_score"],[3,2,1,"","prepare_text"],[3,2,1,"","prime_text"],[3,2,1,"","query"],[3,2,1,"","score"],[3,2,1,"","token_score"],[3,2,1,"","topk"]],"minicons.scorer.MaskedLMScorer":[[3,2,1,"","add_special_tokens"],[3,2,1,"","cloze"],[3,2,1,"","cloze_distribution"],[3,2,1,"","compute_stats"],[3,2,1,"","distribution"],[3,2,1,"","logprobs"],[3,2,1,"","mask"],[3,2,1,"","prepare_text"],[3,2,1,"","prime_text"],[3,2,1,"","sequence_score"],[3,2,1,"","token_score"]],"minicons.supervised":[[4,1,1,"","SupervisedHead"]],"minicons.supervised.SupervisedHead":[[4,2,1,"","encode"],[4,2,1,"","logits"]],"minicons.utils":[[5,3,1,"","argmax"],[5,3,1,"","argmin"],[5,3,1,"","character_span"],[5,3,1,"","edit_distance"],[5,3,1,"","find_index"],[5,3,1,"","find_paired_indices"],[5,3,1,"","find_pattern"],[5,3,1,"","gen_words"],[5,3,1,"","get_batch"],[5,3,1,"","mask"]],minicons:[[2,0,0,"-","cwe"],[3,0,0,"-","scorer"],[4,0,0,"-","supervised"],[5,0,0,"-","utils"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function"},terms:{"0":[2,3,7,8],"0005340576171875":8,"0007704822928644717":8,"0013275146484375":8,"0019151987507939339":8,"0029":7,"0032":7,"0086":7,"0098":7,"0105":7,"0150":7,"0163":7,"0228":7,"0279":7,"0307":7,"0308":7,"0311":7,"0317":7,"0331":7,"0434":7,"0460":7,"0470":7,"0482":7,"0556":7,"0583":7,"0601":7,"0614":7,"0645":7,"0687":7,"0695":7,"0717":7,"0792":7,"0796":7,"0830":7,"0835":7,"0846":7,"0873":7,"0942":7,"0958":7,"1":[2,3,5,7,8],"10":8,"100":8,"1031":7,"1035":7,"1065":7,"1097":7,"11":7,"1103":7,"1106":7,"1151":7,"1154":7,"12":[7,8],"1224":7,"1243":7,"13":7,"1302":7,"1305":7,"1328":7,"1329":7,"1332":7,"1333":7,"1415":7,"1451":7,"15":[7,8],"1533":7,"1559":7,"1582":7,"1607":7,"1613":7,"1674":7,"1677":7,"1710":7,"1711":7,"1713":7,"1723":7,"1725":7,"1799":7,"18":3,"1852":7,"1863":7,"1907":7,"1977":7,"2":[3,7,8],"20":7,"2020":3,"2032":7,"2114":7,"2143":7,"2156":7,"2164":7,"2180":7,"2197":7,"2212":7,"2250":7,"2269":7,"2270":7,"2290":7,"2304":7,"2318":7,"2355":7,"2379":7,"2445":7,"2449":7,"2477":7,"2502":7,"2528":7,"2572":7,"2670":7,"2701":7,"2713":7,"2731":7,"2735":7,"2772":7,"2797":7,"2806":7,"2816":7,"2845":7,"2876":7,"2902":7,"2918":7,"2921":7,"2985":7,"2992":7,"2998":7,"3":8,"3004":7,"3012":7,"3013":7,"3141":7,"3149":7,"3187":7,"3262":7,"3281":7,"3293":7,"3336":7,"3337":7,"3381":7,"3382":7,"3388":7,"3450":7,"3467":7,"3472":7,"3499":7,"3509":7,"3511":7,"3571":7,"3577":7,"3579":7,"3605":7,"3628":7,"3641":7,"3650":7,"37":8,"3723":7,"3773":7,"3776":7,"3800":7,"3833":7,"39":8,"392584800720215":8,"3931":7,"3938":7,"4":7,"4013":7,"4021":7,"4052":7,"4080":7,"4175":7,"4210":7,"4240":7,"4257":7,"4277":7,"4349":7,"4395":7,"4418":7,"4591":7,"4638":7,"4729":7,"4842":7,"4878":7,"4887":7,"4973":7,"5":[7,8],"5046":7,"5105514526367188":8,"5114":7,"5127":7,"525080680847168":8,"5314":7,"5321":7,"5355":7,"5384":7,"550mb":8,"5510":7,"5519":7,"5522":7,"5606":7,"5618":7,"5652":7,"5739":7,"5793":7,"5835":7,"5841":7,"5866":7,"5890":7,"5956":7,"5963":7,"6":8,"6095":7,"6099":7,"612955093383789":8,"621960163116455":8,"6266":7,"6336":7,"6424":7,"6474":7,"6631927490234375":8,"669326782226562":8,"6711":7,"6786":7,"681724548339844":8,"6836":7,"6865":8,"6907":7,"6910":7,"6916":7,"6923":7,"69605827331543":8,"6981":8,"7095":7,"7379":7,"7613":7,"7634":7,"7677":7,"7844":7,"7924":7,"8":[7,8],"8011":7,"8018":7,"8026":7,"804":8,"8121":7,"8467":7,"8616":7,"8634":7,"8643":7,"8667":7,"8791":7,"879678726196289":8,"8803":7,"9":8,"9128":7,"9216":7,"9296":7,"929980278015137":8,"9310":7,"9413":7,"9539":7,"9541":7,"962379455566406":8,"9663":7,"9704":7,"9946":7,"abstract":3,"case":3,"class":[2,3,4,7],"default":[2,3,4,7,8],"do":[7,8],"final":8,"float":3,"function":[0,3,7,8],"import":[3,5,8],"int":[2,3,5],"new":0,"return":[2,3,4,8],"true":[2,3,4,8],A:[0,3,7,8],By:[3,7,8],For:[3,7,8],If:3,In:0,It:[7,8],One:7,The:[3,7,8],Then:0,These:7,To:8,about:8,accept:[3,7,8],across:0,ad:0,adapt_scor:3,add:[3,8],add_special_token:3,addit:8,agreement:8,aircraft:7,al:3,algorithm:3,allow:[7,8],along:3,alreadi:8,also:[0,3,7,8],altern:0,alwai:3,an:[0,2,3,4,7,8],analys:4,analysi:3,ani:[0,3,7],append:[7,8],ar:[2,3,7],arg:3,argmax:5,argmin:5,argument:8,around:7,articl:8,assign:8,associ:3,attribut:7,automodel:7,autoregress:3,avail:[0,7],averag:7,avg:3,b:8,bad:8,bad_scor:8,base:[2,3,4,7,8],base_two:[3,8],batch:[0,2,3,4,7,8],batch_siz:[5,8],behavior:[4,8],bert:[3,7],bike:7,bin:[2,3],binari:0,bit:[3,8],blank:3,blimp:8,book:7,bool:[2,3,4,5,8],borgia:8,borrow:3,both:0,brief:8,built:3,bw_i:3,calcul:[0,3],call:3,callabl:3,can:[0,3,7,8],chang:3,charact:[2,7],character_span:[5,7],check:3,choos:7,clone:0,cloze:3,cloze_distribut:3,code:[3,7,8],com:[0,3],command:0,compar:8,comput:[0,3,8],compute_stat:[3,8],condit:[3,8],conduct:0,consist:[2,3,4,7],constitu:7,construct:4,contain:[3,8],content:6,context:[0,3,5,8],contextu:[0,2,7],continu:3,contist:3,conveni:3,convent:8,convert:8,core:0,correspond:[2,7],counterpart:3,cpu:[0,2,3,4,7,8],cuda:[2,3,7],cwe:[0,1,6,7],data:[5,7,8],dataload:8,dataset:8,deal:8,decod:3,demonstr:[7,8],depend:[3,4],deprec:3,descript:0,desir:3,detail:0,devic:[2,3,4,7],dict:[3,4],dictionari:4,differ:[0,7],directori:7,distractor_agreement_relational_noun:8,distribut:3,divid:8,document:8,doe:[3,7,8],doesn:8,don:8,done:3,download:8,e:[0,3,7],each:[3,4,7,8],easier:7,edit:0,edit_dist:5,effici:[0,7],either:[2,3,4,7],element:3,embed:[0,2,7],encod:[2,3,4,8],encode_text:2,end:[2,7],end_1:7,end_2:7,end_i:7,end_n:7,ensur:3,environ:0,equival:3,estim:3,et:3,etc:3,evalu:8,even:7,everi:[3,7],exampl:[7,8],exist:4,experi:8,explor:0,extend:8,extent:8,extract:[0,2],extract_paired_represent:2,extract_represent:[2,7],extract_sentence_represent:2,extrem:0,f:[7,8],facilit:[2,3,4,8],fals:[2,3,4,5],file:[0,2,3,7,8],fin:7,find_index:5,find_paired_indic:5,find_pattern:5,first:[7,8],fit:3,flat:0,flaticon:0,follow:[3,7,8],form:[2,3],format:[3,4,7,8],from:[0,2,3,7,8],full:8,g:[0,3,8],gen_word:5,gener:0,get:[3,7,8],get_batch:5,git:0,github:[0,3],given:[2,3,8],good:8,good_scor:8,gpt2:[3,8],gpt:8,gpu:[0,7],grab:0,greater:8,guest:8,ha:[0,3,8],handi:0,handl:0,hasn:8,have:[0,3,8],haven:8,head:4,help:8,here:[0,8],hi:7,hidden:2,hidden_st:2,how:[3,8],http:3,hub:[0,2,3,7],huggingfac:[0,2,3,7,8],hypothesi:8,i:[7,8],icon:0,id:3,idx:3,ignor:3,implement:[2,4],increment:[3,8],incrementallmscor:[3,8],inde:7,index:7,indic:[2,7],infer:[3,4],initi:2,input:[2,3,4,7,8],input_id:2,instal:0,instanc:7,instanti:8,instead:3,intend:3,intuit:7,involv:7,item:3,iter:[3,8],ith:7,its:3,jet:7,json:8,jsonl:8,k:3,kanishkamisra:0,kind:3,kwarg:3,label:4,lambda:[3,8],langaug:3,languag:[0,3],larg:[0,8],last:7,last_four:7,layer:[0,2,7],left:[3,8],len:7,length:[3,5,8],let:[7,8],level:0,light:8,like:[0,7],line:[0,7,8],list:[2,3,4,5,7,8],lm:3,lmscorer:3,load:[2,3,8],local:[2,3],log:[0,3,8],logic:2,logit:4,logo:0,logprob:[3,8],lst:5,made:[0,7],main:3,make:[0,7],manual:3,manual_speci:3,mask:[2,3,5],maskedlmscor:3,match:3,max:2,mean:[2,3,7,8],meaning:3,measur:0,method:[3,5,7],metric:3,min:2,mlm:3,model:[0,2,3,4],model_nam:[2,3,4],modif:3,modifi:3,modul:[0,6,7],more:[2,8],most:8,multipl:7,my:7,name:[2,3,4],natur:3,newspap:8,next:3,next_word_distribut:3,niec:8,none:[2,3],normal:8,note:[3,8],now:7,np:8,number:[7,8],numpi:8,object:[2,3,4],one:[2,8],one_prefix_prefix:8,one_prefix_word_bad:8,one_prefix_word_good:8,onli:[3,8],open:[7,8],option:[2,3,4,5],other:3,our:[7,8],out:[2,3],output:[3,4,8],over:3,overal:[3,8],p:[3,8],packag:[0,3,6],pair:8,paramet:[2,3,4,8],part:3,partial_scor:3,pass:[3,4,7,8],passion:7,path:[2,3],pattern:8,per:[0,3],perform:[0,8],phrase:0,piec:5,pip:0,poetri:0,pool:[2,3,7],pooler:2,posit:3,preambl:3,prefer:3,prefix:3,prepar:[3,7],prepare_text:[3,8],preprocess:3,present:[2,3],pretrain:[2,3],primari:3,primarili:7,prime:3,prime_text:3,print:8,prob:[3,4,8],probabili:3,probabl:[0,3,4,8],process:3,prod:3,prompt:3,propuls:7,provid:[3,7],pt:[2,3,4],purpos:[7,8],queri:3,r:[7,8],randomli:2,rang:7,rank:[3,8],raw:2,read:7,reduc:7,reduct:[3,8],reformat:3,regardless:7,regular:5,report:3,repositori:3,repres:[3,7,8],represent:[0,2],result:3,return_tensor:[3,4,8],reveal:4,ride:7,right:3,roberta:3,row:8,run:[0,3,4,8],s:[2,3,7,8],s_i:3,salazar:3,same:[3,7],samplesent:7,scale:[0,8],score:[0,3,8],scorer:[0,1,6,8],second:[7,8],see:8,select:3,senat:8,sentenc:[0,2,3,5,8],sentence_1:7,sentence_2:7,sentence_n:7,sentence_word:[2,3],sequenc:3,sequence_scor:[3,8],shell:0,ship:0,should:[2,3,4,8],show:8,shuffl:5,simonepri:3,simonpri:3,singl:[0,3,8],sketch:8,small:[7,8],so:[7,8],some:3,soon:0,sourc:0,span:[2,7],special:3,specif:[3,8],specifi:[3,4,7],stack:7,stand:[0,3,7],start:[2,7],start_1:7,start_2:7,start_i:7,start_n:7,state:2,stimuli:[3,8],stimuli_dl:8,store:[2,3],str:[2,3,4,5],string:3,strip:7,submodul:[0,6],sum:8,supervis:[0,1,6],supervisedhead:4,sure:0,surpris:[0,3],t:8,target:7,task:[4,8],tensor:[2,3,4,7,8],test:8,text:[2,3,8],than:8,thei:[4,7],theori:7,therefor:8,thi:[3,7,8],those:8,todo:3,token:[3,8],token_scor:[3,8],tool:0,topk:3,torch:[2,3,4,7,8],train:4,transform:[2,3,7],trick:7,truck:8,tupl:[2,3,5],tutori:7,two:0,txt:7,type:[2,3,7],uncas:7,under:4,union:[2,3,4],us:[0,2,3],usual:3,util:[0,1,6,7,8],v:3,valu:[3,8],verbos:4,virtual:0,vocab:3,vocabulari:3,vs:8,wa:4,want:[3,8],warn:8,we:7,weight:2,what:[3,7],when:3,where:[0,2,3,7],whether:[2,3,4,8],which:[2,3,4,7,8],whole:5,whose:3,word1:5,word2:5,word:[0,2,3,5,8],word_1:7,word_2:7,word_n:7,work:[3,7],would:[0,8],wrapper:7,x:[3,8],yield:7,you:[0,3,8],zip:8},titles:["minicons: flexible behavioral analyses of transformer LMs","minicons package","minicons.cwe module","minicons.scorer module","minicons.supervised module","minicons.utils module","minicons","Extracting Word and Phrase Representations using minicons","Calculating surprisals with transformer models using minicons"],titleterms:{"import":7,analys:0,behavior:0,calcul:8,content:[0,1],cwe:2,exampl:0,extract:7,flexibl:0,get:0,introduct:0,librari:7,lm:0,load:7,minicon:[0,1,2,3,4,5,6,7,8],model:[7,8],modul:[1,2,3,4,5],packag:[1,7],phrase:7,preliminari:7,represent:7,reprsent:7,requir:7,scorer:3,sentenc:7,start:0,submodul:1,supervis:4,surpris:8,transform:[0,8],us:[7,8],util:5,word:7}})