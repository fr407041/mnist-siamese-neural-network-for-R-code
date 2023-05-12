# MNIST Siamese Neural Network for Keras in R

I have been eagerly waiting for R Keras to release an example code for the mnist_siamese_graph.R model, but it seems that there has been no progress. Hence, I decided to take the initiative and complete this job myself.

You can check out my Kaggle kernel introduction [here](https://github.com/rstudio/keras/blob/master/vignettes/examples/mnist_siamese_graph.R), where I hope you will learn the following four things:

1. Building a self-defined generator using R Keras, which has few related materials available online.
1. Building a self-defined layer using R Keras (as above).
1. Building a self-defined backend function using R Keras (as above).
1. Gaining some knowledge about the Siamese neural network.

Now we start to build siamese neural network for mnist number's similarity.

**Step 1 : Load the Data**
<br>First, we need to load and split the data into training, validation, and testing sets. We'll use the Keras and abind libraries in R to do this. Here's the code:
```
library(keras)
library(abind)
mnist <- dataset_mnist("/kaggle/input/mnist.npz")
train_images <- mnist$train$x
train_labels <- mnist$train$y
set.seed(9478) 
val_idx      <- sample( 1:nrow(train_images) , size = ceiling(0.2*nrow(train_images)) , replace = F ) 
val_images   <- train_images[val_idx,,] 
val_labels   <- train_labels[val_idx]  
train_images <- train_images[-val_idx,,] 
train_labels <- train_labels[-val_idx]  
test_images  <- mnist$test$x 
test_labels  <- mnist$test$y  
```

You can see what the MNIST data looks like below. The label corresponds to the number in the image. Note that the array for each sample has been divided by 255, so the maximum value is 1 (white) and the minimum value is 0 (black).

Even if you're not a technical reader, don't worry! This guide will provide an accessible introduction to building a siamese neural network.

```
par(mar = c(0,0,4,0))
i <- 1
plot( as.raster(train_images[i,,]/255) )
title( train_labels[i] )
```
![number](https://github.com/fr407041/mnist-siamese-neural-network-for-R-code/blob/master/images/plot_number.png)

**Step 2: Setting Parameters**
```
num_classes  <- 10 # mnist dataset only contains numbers: 0,1,2,3,4,5,6,7,8,9 
shape_size   <- 28 # mnist images have a shape of (28,28)
train_batch  <- 20 
val_batch    <- 20 
```
**Step 3: Data Preprocessing**
<br>We create two lists: **`train_data_list`** and **`val_data_list`** to hold the training and validation data. We also group the data by their labels and create sublists of images and labels.
```
train_data_list    <- list() 
grp_kind     <- sort( unique( train_labels ) )   
  for( grp_idx in 1:length(grp_kind) ) { # grp_idx = 1     
    label                      <- grp_kind[grp_idx]     
    tmp_images                 <- train_images[train_labels==label,,]     
    tmp_images                 <- array( tmp_images , dim = c( dim(tmp_images) , 1) )  # why reshape array? because keras image_data_generator only accept rank = 4  
    train_data_list[[grp_idx]] <- list( data  = tmp_images ,                                         
                                        label = train_labels[train_labels==label]                                        
                                      )   
  }  

val_data_list      <- list() 
grp_kind     <- sort( unique( val_labels ) )   
  for( grp_idx in 1:length(grp_kind) ) { # grp_idx = 1     
    label                      <- grp_kind[grp_idx]     
    tmp_images                 <- val_images[val_labels==label,,]     
    tmp_images                 <- array( tmp_images , dim = c( dim(tmp_images) , 1) )     
    val_data_list[[grp_idx]]   <- list( data  = tmp_images ,                                         
                                        label = val_labels[val_labels==label]      
                                      )   
  }
```
  **Step 4: Building Self-Defined Generator**

<br> We create a separate generator for each number in the dataset, allowing us to utilize the data augmentation provided by the image_data_generator to generalize our data. Next, we create a list to collect all the generators for each number, and use this list as input to our self-defined generator, called join_generator. This generator uses the list of generators as a parameter to build balanced datasets of images with the same number and different numbers. 
 
(Note: Each generator has a batch size of 1, which will allow us to build a self-defined generator.)

```
train_datagen = image_data_generator(
  rescale = 1/255          ,
  rotation_range = 5       ,
  width_shift_range = 0.1  ,
  height_shift_range = 0.05,
  #shear_range = 0.1,
  zoom_range = 0.1         ,
  horizontal_flip = FALSE  ,
  vertical_flip = FALSE    ,
  fill_mode = "constant"
)

train_0_generator <- flow_images_from_data( # for 0 number
  x = train_data_list[[1]]$data  ,   
  y = train_data_list[[1]]$label ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 9487                    ,   
  batch_size = 1 
)

train_1_generator <- flow_images_from_data( # for 1 number  
  x = train_data_list[[2]]$data  ,   
  y = train_data_list[[2]]$label ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 9487                    ,   
  batch_size = 1 
)

train_2_generator <- flow_images_from_data( # for 2 number  
  x = train_data_list[[3]]$data  ,   
  y = train_data_list[[3]]$label ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 9487                    ,   
  batch_size = 1 
)

train_3_generator <- flow_images_from_data( # for 3 number  
  x = train_data_list[[4]]$data  ,   
  y = train_data_list[[4]]$label ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 9487                    ,   
  batch_size = 1 
)

train_4_generator <- flow_images_from_data( # for 4 number  
  x = train_data_list[[5]]$data  ,   
  y = train_data_list[[5]]$label ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 9487                    ,   
  batch_size = 1 
)

train_5_generator <- flow_images_from_data( # for 5 number  
  x = train_data_list[[6]]$data  ,   
  y = train_data_list[[6]]$label ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 9487                    ,   
  batch_size = 1 
)

train_6_generator <- flow_images_from_data( # for 6 number  
  x = train_data_list[[7]]$data  ,   
  y = train_data_list[[7]]$label ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 9487                    ,   
  batch_size = 1 
)

train_7_generator <- flow_images_from_data( # for 7 number  
  x = train_data_list[[8]]$data  ,   
  y = train_data_list[[8]]$label ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 9487                    ,   
  batch_size = 1 
)

train_8_generator <- flow_images_from_data( # for 8 number   
  x = train_data_list[[9]]$data  ,   
  y = train_data_list[[9]]$label ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 9487                    ,   
  batch_size = 1 
)

train_9_generator <- flow_images_from_data( # for 9 number   
  x = train_data_list[[10]]$data ,   
  y = train_data_list[[10]]$label,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 9487                    ,   
  batch_size = 1 
)

val_test_datagen = image_data_generator(
  rescale = 1/255
)

val_0_generator <- flow_images_from_data( # for 0 number  
  x = val_data_list[[1]]$data    ,   
  y = val_data_list[[1]]$label   ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 9487                    ,   
  batch_size = 1 
)

val_1_generator <- flow_images_from_data( # for 1 number  
  x = val_data_list[[2]]$data    ,   
  y = val_data_list[[2]]$label   ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 9487                    ,   
  batch_size = 1 
)

val_2_generator <- flow_images_from_data( # for 2 number   
  x = val_data_list[[3]]$data    ,   
  y = val_data_list[[3]]$label   ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 9487                    ,   
  batch_size = 1 
)

val_3_generator <- flow_images_from_data( # for 3 number   
  x = val_data_list[[4]]$data    ,   
  y = val_data_list[[4]]$label   ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 9487                    ,   
  batch_size = 1 
)

val_4_generator <- flow_images_from_data( # for 4 number   
  x = val_data_list[[5]]$data    ,   
  y = val_data_list[[5]]$label   ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 9487                    ,   
  batch_size = 1 
)

val_5_generator <- flow_images_from_data( # for 5 number  
  x = val_data_list[[6]]$data    ,   
  y = val_data_list[[6]]$label   ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 9487                    ,   
  batch_size = 1 
)

val_6_generator <- flow_images_from_data( # for 6 number  
  x = val_data_list[[7]]$data    ,   
  y = val_data_list[[7]]$label   ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 9487                    ,   
  batch_size = 1 
)

val_7_generator <- flow_images_from_data( # for 7 number  
  x = val_data_list[[8]]$data    ,   
  y = val_data_list[[8]]$label   ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 9487                    ,   
  batch_size = 1 
)

val_8_generator <- flow_images_from_data( # for 8 number  
  x = val_data_list[[9]]$data    ,   
  y = val_data_list[[9]]$label   ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 9487                    ,   
  batch_size = 1 
)

val_9_generator <- flow_images_from_data( # for 9 number  
  x = val_data_list[[10]]$data   ,   
  y = val_data_list[[10]]$label  ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 9487                    ,   
  batch_size = 1 
)

train_generator_list <- list(
  train_0_generator ,
  train_1_generator ,
  train_2_generator ,
  train_3_generator ,
  train_4_generator ,
  train_5_generator ,
  train_6_generator ,
  train_7_generator ,
  train_8_generator ,
  train_9_generator 
)

val_generator_list <- list(
  val_0_generator   ,
  val_1_generator   ,
  val_2_generator   ,
  val_3_generator   ,
  val_4_generator   ,
  val_5_generator   ,
  val_6_generator   ,
  val_7_generator   ,
  val_8_generator   ,
  val_9_generator 
)

join_generator <- function( generator_list , batch ) { 
  function() { 
    batch_left  <- NULL 
    batch_right <- NULL 
    similarity  <- NULL 
      for( i in seq_len(batch) ) { # i = 1 
          # front half 
          if( i <= ceiling(batch/2) ) { # It's suggest to use balance of positive and negative data set, so I divide half is 1(same) and another is 0(differnet).
            grp_same    <- sample( seq_len(num_classes) , 1 ) 
            batch_left  <- abind( batch_left , generator_next(generator_list[[grp_same]])[[1]] , along = 1 ) 
            batch_right <- abind( batch_right , generator_next(generator_list[[grp_same]])[[1]] , along = 1 ) 
            similarity  <- c( similarity , 1 ) # 1 : from the same number
            #par(mar = c(0,0,4,0)) 
            #plot( as.raster(batch_left[21,,,]) ) 
            #title( batch_left[[2]] ) 
          } else { # after half 
            grp_diff    <- sort( sample( seq_len(num_classes) , 2 ) ) 
            batch_left  <- abind( batch_left , generator_next(generator_list[[grp_diff[1]]])[[1]] , along = 1 ) 
            batch_right <- abind( batch_right , generator_next(generator_list[[grp_diff[2]]])[[1]] , along = 1 ) 
            similarity  <- c( similarity , 0 ) # 0 : from the differnet number
          } 
      } 
    return( list( list( batch_left , batch_right ), similarity ) ) 
  } 
}

train_join_generator   <- join_generator( train_generator_list , train_batch )
val_join_generator     <- join_generator( val_generator_list   , val_batch   )
```
**Step 5: Building the Siamese model**
<br>Now we can build our siamese neural network for recognizing similarity between two MNIST numbers. We first create a simple convolutional neural network model conv_base, which will be shared by two input images with the same weights.
![siamese](https://github.com/fr407041/mnist-siamese-neural-network-for-R-code/blob/master/images/saimese_model.jpg)
```
left_input_tensor      <- layer_input(shape = list(shape_size, shape_size, 1), name = "left_input_tensor")
right_input_tensor     <- layer_input(shape = list(shape_size, shape_size, 1), name = "right_input_tensor")

conv_base              <- keras_model_sequential()           %>%
  layer_flatten(input_shape=list(shape_size, shape_size, 1)) %>%
  layer_dense(units = 128, activation = "relu", name='fc1')  %>%
  layer_dropout(rate = 0.1, name='dropout1')                 %>%
  layer_dense(units = 128, activation = "relu", name='fc2')  %>% 
  layer_dropout(rate = 0.1, name='dropout2')                 %>%
  layer_dense(units = 128, activation = "relu", name='fc3')

left_output_tensor     <- left_input_tensor  %>%   
                          conv_base  

right_output_tensor    <- right_input_tensor %>%   
                          conv_base  

L1_distance <- function(tensors) { # build keras backend's function  
  c(x,y) %<-% tensors   
  return( k_abs( x - y ) ) 
}       

L1_layer    <- layer_lambda( object = list(left_output_tensor,right_output_tensor) , # To build self define layer, you must use layer_lamda                                
                             f = L1_distance                              
                           )   

prediction  <- L1_layer%>%                
               layer_dense( units = 1 , activation = "sigmoid" )  

model       <- keras_model( list(left_input_tensor,right_input_tensor), prediction)

**Step 6: Model Fit**
<br> Notice! Small learning rate (ex : 1e-5) will lead to slowly optimize progress, so we suggest use 1e-3 as our learning rate.
<br>

model %>% compile(
  loss      = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-3),
  metrics   = c("accuracy")
)

history <- model %>% fit_generator(
  train_join_generator,
  steps_per_epoch = 100,
  epochs = 400,
  validation_data = val_join_generator,
  validation_steps = 50
)
```
**Step 6: Train the Model**
<br>Note: Using a small learning rate (e.g. 1e-5) can result in slower optimization progress. Therefore, we recommend using a learning rate of 1e-3 to speed up the training process.
<br>As shown below, the model trained with a learning rate of 1e-3 learns faster than the one trained with a learning rate of 1e-5.
![learning_rate_comparison](https://github.com/fr407041/mnist-siamese-neural-network-for-R-code/blob/master/images/learning_Rate_comparison.png)
```
model %>% compile(
  loss      = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-3),
  metrics   = c("accuracy")
)

history <- model %>% fit_generator(
  train_join_generator,
  steps_per_epoch = 100,
  epochs = 50,
  validation_data = val_join_generator,
  validation_steps = 50
)

plot(history)
```
**Step 7: Test Data Verified**
<br> Same number match
```
# same number
mnist_number_left  <- 8
filter_idx_left    <- sample( which( test_labels == mnist_number_left  ) , 1 )
img_input_left     <- test_images[filter_idx_left ,,]/255
mnist_number_right <- 8
filter_idx_right   <- sample( which( test_labels == mnist_number_right ) , 1 )
img_input_right    <- test_images[filter_idx_right,,]/255
img_input_left     <- array_reshape(img_input_left , c(1, shape_size, shape_size, 1))
img_input_right    <- array_reshape(img_input_right, c(1, shape_size, shape_size, 1))

similarity         <- model %>% predict(list(img_input_left,img_input_right))
par(mar = c(0,0,4,0))
plot( as.raster( abind(img_input_left[1,,,] ,
                       img_input_right[1,,,],
                       along = 2
                      ) 
               ) 
)
title( paste0( test_labels[filter_idx_left] , " v.s " , test_labels[filter_idx_right] , " , similarity : " , round(similarity,3) ) )
```
![learning_rate_comparison](https://github.com/fr407041/mnist-siamese-neural-network-for-R-code/blob/master/images/same_comparison.png)

Different number match
```
# different number
mnist_number_left  <- 8
filter_idx_left    <- sample( which( test_labels == mnist_number_left  ) , 1 )
img_input_left     <- test_images[filter_idx_left ,,]/255
mnist_number_right <- 7
filter_idx_right   <- sample( which( test_labels == mnist_number_right ) , 1 )
img_input_right    <- test_images[filter_idx_right,,]/255
img_input_left     <- array_reshape(img_input_left , c(1, shape_size, shape_size, 1))
img_input_right    <- array_reshape(img_input_right, c(1, shape_size, shape_size, 1))

similarity         <- model %>% predict(list(img_input_left,img_input_right))
par(mar = c(0,0,4,0))
plot( as.raster( abind(img_input_left[1,,,] ,
                       img_input_right[1,,,],
                       along = 2
                      ) 
               ) 
    )
title( paste0( test_labels[filter_idx_left] , " v.s " , test_labels[filter_idx_right] , " , similarity : " , round(similarity,3) ) )
```
![learning_rate_comparison](https://github.com/fr407041/mnist-siamese-neural-network-for-R-code/blob/master/images/different_comparison.png)
