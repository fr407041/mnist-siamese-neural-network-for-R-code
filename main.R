library(keras)
library(abind)
mnist <- dataset_mnist()
train_images <- mnist$train$x
train_labels <- mnist$train$y

set.seed(911) 
val_idx      <- sample( 1:nrow(train_images) , size = ceiling(0.2*nrow(train_images)) , replace = F ) 
val_images   <- train_images[val_idx,,] 
val_labels   <- train_labels[val_idx]  
train_images <- train_images[-val_idx,,] 
train_labels <- train_labels[-val_idx]  
test_images  <- mnist$test$x 
test_labels  <- mnist$test$y  
input_shape  <- dim(train_images)[2:3]  
num_classes  <- 10 
shape_size   <- 28 
train_batch  <- 20 
val_batch    <- 20 


train_data_list    <- list() 
grp_kind     <- sort( unique( train_labels ) )   
  for( grp_idx in 1:length(grp_kind) ) { # grp_idx = 1     
    label                      <- grp_kind[grp_idx]     
    tmp_images                 <- train_images[train_labels==label,,]     
    tmp_images                 <- array( tmp_images , dim = c( dim(tmp_images) , 1) )     
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



##### generator #####
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

train_0_generator <- flow_images_from_data(   
  x = train_data_list[[1]]$data  ,   
  y = train_data_list[[1]]$label ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 911                     ,   
  batch_size = 1 
)

train_1_generator <- flow_images_from_data(   
  x = train_data_list[[2]]$data  ,   
  y = train_data_list[[2]]$label ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 911                     ,   
  batch_size = 1 
)

train_2_generator <- flow_images_from_data(   
  x = train_data_list[[3]]$data  ,   
  y = train_data_list[[3]]$label ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 911                     ,   
  batch_size = 1 
)

train_3_generator <- flow_images_from_data(   
  x = train_data_list[[4]]$data  ,   
  y = train_data_list[[4]]$label ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 911                     ,   
  batch_size = 1 
)

train_4_generator <- flow_images_from_data(   
  x = train_data_list[[5]]$data  ,   
  y = train_data_list[[5]]$label ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 911                     ,   
  batch_size = 1 
)

train_5_generator <- flow_images_from_data(   
  x = train_data_list[[6]]$data  ,   
  y = train_data_list[[6]]$label ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 911                     ,   
  batch_size = 1 
)

train_6_generator <- flow_images_from_data(   
  x = train_data_list[[7]]$data  ,   
  y = train_data_list[[7]]$label ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 911                     ,   
  batch_size = 1 
)

train_7_generator <- flow_images_from_data(   
  x = train_data_list[[8]]$data  ,   
  y = train_data_list[[8]]$label ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 911                     ,   
  batch_size = 1 
)

train_8_generator <- flow_images_from_data(   
  x = train_data_list[[9]]$data  ,   
  y = train_data_list[[9]]$label ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 911                     ,   
  batch_size = 1 
)

train_9_generator <- flow_images_from_data(   
  x = train_data_list[[10]]$data ,   
  y = train_data_list[[10]]$label,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 911                     ,   
  batch_size = 1 
)

val_test_datagen = image_data_generator(
  rescale = 1/255
)

val_0_generator <- flow_images_from_data(   
  x = val_data_list[[1]]$data    ,   
  y = val_data_list[[1]]$label   ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 911                     ,   
  batch_size = 1 
)

val_1_generator <- flow_images_from_data(   
  x = val_data_list[[2]]$data    ,   
  y = val_data_list[[2]]$label   ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 911                     ,   
  batch_size = 1 
)

val_2_generator <- flow_images_from_data(   
  x = val_data_list[[3]]$data    ,   
  y = val_data_list[[3]]$label   ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 911                     ,   
  batch_size = 1 
)

val_3_generator <- flow_images_from_data(   
  x = val_data_list[[4]]$data    ,   
  y = val_data_list[[4]]$label   ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 911                     ,   
  batch_size = 1 
)

val_4_generator <- flow_images_from_data(   
  x = val_data_list[[5]]$data    ,   
  y = val_data_list[[5]]$label   ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 911                     ,   
  batch_size = 1 
)

val_5_generator <- flow_images_from_data(   
  x = val_data_list[[6]]$data    ,   
  y = val_data_list[[6]]$label   ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 911                     ,   
  batch_size = 1 
)

val_6_generator <- flow_images_from_data(   
  x = val_data_list[[7]]$data    ,   
  y = val_data_list[[7]]$label   ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 911                     ,   
  batch_size = 1 
)

val_7_generator <- flow_images_from_data(   
  x = val_data_list[[8]]$data    ,   
  y = val_data_list[[8]]$label   ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 911                     ,   
  batch_size = 1 
)

val_8_generator <- flow_images_from_data(   
  x = val_data_list[[9]]$data    ,   
  y = val_data_list[[9]]$label   ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 911                     ,   
  batch_size = 1 
)

val_9_generator <- flow_images_from_data(   
  x = val_data_list[[10]]$data   ,   
  y = val_data_list[[10]]$label  ,   
  train_datagen                  ,   
  shuffle = TRUE                 ,   
  seed = 911                     ,   
  batch_size = 1 
)
#####

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
          if( i <= ceiling(batch/2) ) { 
            grp_same    <- sample( seq_len(num_classes) , 1 ) 
            batch_left  <- abind( batch_left , generator_next(generator_list[[grp_same]])[[1]] , along = 1 ) 
            batch_right <- abind( batch_right , generator_next(generator_list[[grp_same]])[[1]] , along = 1 ) 
            similarity  <- c( similarity , 1 ) 
            #par(mar = c(0,0,4,0)) 
            #plot( as.raster(batch_left[21,,,]) ) 
            #title( batch_left[[2]] ) 
          } else { # after half 
            grp_diff    <- sort( sample( seq_len(num_classes) , 2 ) ) 
            batch_left  <- abind( batch_left , generator_next(generator_list[[grp_diff[1]]])[[1]] , along = 1 ) 
            batch_right <- abind( batch_right , generator_next(generator_list[[grp_diff[2]]])[[1]] , along = 1 ) 
            similarity  <- c( similarity , 0 ) 
          } 
      } 
    return( list( list( batch_left , batch_right ), similarity ) ) 
  } 
}

train_join_generator   <- join_generator( train_generator_list , train_batch )
val_join_generator     <- join_generator( val_generator_list   , val_batch   )


left_input_tensor      <- layer_input(shape = list(shape_size, shape_size, 1), name = "left_input_tensor")
right_input_tensor     <- layer_input(shape = list(shape_size, shape_size, 1), name = "right_input_tensor")

conv_base              <- keras_model_sequential() %>%
  layer_flatten(input_shape=list(shape_size, shape_size, 1)) %>%
  layer_dense(units = 128, activation = "relu", name='fc1')  %>%
  layer_dropout(rate = 0.1, name='dropout1')                 %>%
  layer_dense(units = 128, activation = "relu", name='fc2')  %>% 
  layer_dropout(rate = 0.1, name='dropout2')                 %>%
  layer_dense(units = 128, activation = "relu", name='fc3')

left_output_tensor     <- left_input_tensor %>%   
                          conv_base  

right_output_tensor    <- right_input_tensor %>%   
                          conv_base  

L1_distance <- function(tensors) {   
  c(x,y) %<-% tensors   
  return( k_abs( x - y ) ) 
}       

L1_layer    <- layer_lambda( object = list(left_output_tensor,right_output_tensor) ,                               
                             f = L1_distance                              
                           )   

prediction  <- L1_layer%>%                
               layer_dense( units = 1 , activation = "sigmoid" )  

model       <- keras_model( list(left_input_tensor,right_input_tensor), prediction)




model %>% compile(
  loss      = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-3),
  metrics   = c("accuracy")
)


history <- model %>% fit_generator(
  train_join_generator                ,
  steps_per_epoch = 100               ,
  epochs = 20                         ,
  validation_data = val_join_generator,
  validation_steps = 50
)


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

