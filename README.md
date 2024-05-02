**The AIC_BIC.R file is used to produce AIC and BIC scores for each model and dataset using mixed-effects modeling. The Discounting_NLMER_delayout file is used to perform LOOCV, holding out a delay at each iteration. 
The two files produce the scores resulting in the following figures:**

<img src="https://github.com/bailejor/DiscountLOOCV/assets/4589448/6eb1641f-8c95-451e-b60e-f07f6843c123" width="300" height="600"> <img src="https://github.com/bailejor/DiscountLOOCV/assets/4589448/277d0eeb-ae00-43b8-9dcc-d83d75fcf535" width="300" height="600"> <img src="https://github.com/bailejor/DiscountLOOCV/assets/4589448/e12a82bd-862c-40ad-a0b9-29605cf6d594" width="300" height="600">

Note. AIC is left panel, BIC is middle panel, and cross-validation is right panel.


**The Discounting_NLMER_delayout_resids.R file produces residuals at each fold of cross-validation and results in the following figure:**

![LOOCV_Resids (-1 to 1)](https://github.com/bailejor/DiscountLOOCV/assets/4589448/5945b930-fcae-44d5-b967-6aa5a50d6a7f)

