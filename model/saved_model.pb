®Ó
Ì£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-0-gb36436b0878ê×

batch_normalization_143/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_143/gamma

1batch_normalization_143/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_143/gamma*
_output_shapes
:*
dtype0

batch_normalization_143/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_143/beta

0batch_normalization_143/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_143/beta*
_output_shapes
:*
dtype0

#batch_normalization_143/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_143/moving_mean

7batch_normalization_143/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_143/moving_mean*
_output_shapes
:*
dtype0
¦
'batch_normalization_143/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_143/moving_variance

;batch_normalization_143/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_143/moving_variance*
_output_shapes
:*
dtype0

conv_layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameconv_layer1/kernel

&conv_layer1/kernel/Read/ReadVariableOpReadVariableOpconv_layer1/kernel*&
_output_shapes
: *
dtype0
x
conv_layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv_layer1/bias
q
$conv_layer1/bias/Read/ReadVariableOpReadVariableOpconv_layer1/bias*
_output_shapes
: *
dtype0

batch_normalization_144/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namebatch_normalization_144/gamma

1batch_normalization_144/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_144/gamma*
_output_shapes
: *
dtype0

batch_normalization_144/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_144/beta

0batch_normalization_144/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_144/beta*
_output_shapes
: *
dtype0

#batch_normalization_144/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization_144/moving_mean

7batch_normalization_144/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_144/moving_mean*
_output_shapes
: *
dtype0
¦
'batch_normalization_144/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'batch_normalization_144/moving_variance

;batch_normalization_144/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_144/moving_variance*
_output_shapes
: *
dtype0

conv_layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*#
shared_nameconv_layer2/kernel

&conv_layer2/kernel/Read/ReadVariableOpReadVariableOpconv_layer2/kernel*&
_output_shapes
: @*
dtype0
x
conv_layer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv_layer2/bias
q
$conv_layer2/bias/Read/ReadVariableOpReadVariableOpconv_layer2/bias*
_output_shapes
:@*
dtype0

batch_normalization_145/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_namebatch_normalization_145/gamma

1batch_normalization_145/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_145/gamma*
_output_shapes
:@*
dtype0

batch_normalization_145/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_145/beta

0batch_normalization_145/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_145/beta*
_output_shapes
:@*
dtype0

#batch_normalization_145/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#batch_normalization_145/moving_mean

7batch_normalization_145/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_145/moving_mean*
_output_shapes
:@*
dtype0
¦
'batch_normalization_145/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'batch_normalization_145/moving_variance

;batch_normalization_145/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_145/moving_variance*
_output_shapes
:@*
dtype0

conv_layer3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameconv_layer3/kernel

&conv_layer3/kernel/Read/ReadVariableOpReadVariableOpconv_layer3/kernel*'
_output_shapes
:@*
dtype0
y
conv_layer3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv_layer3/bias
r
$conv_layer3/bias/Read/ReadVariableOpReadVariableOpconv_layer3/bias*
_output_shapes	
:*
dtype0

batch_normalization_146/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_146/gamma

1batch_normalization_146/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_146/gamma*
_output_shapes	
:*
dtype0

batch_normalization_146/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_146/beta

0batch_normalization_146/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_146/beta*
_output_shapes	
:*
dtype0

#batch_normalization_146/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_146/moving_mean

7batch_normalization_146/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_146/moving_mean*
_output_shapes	
:*
dtype0
§
'batch_normalization_146/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_146/moving_variance
 
;batch_normalization_146/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_146/moving_variance*
_output_shapes	
:*
dtype0

conv_layer4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameconv_layer4/kernel

&conv_layer4/kernel/Read/ReadVariableOpReadVariableOpconv_layer4/kernel*(
_output_shapes
:*
dtype0
y
conv_layer4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv_layer4/bias
r
$conv_layer4/bias/Read/ReadVariableOpReadVariableOpconv_layer4/bias*
_output_shapes	
:*
dtype0

batch_normalization_147/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_147/gamma

1batch_normalization_147/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_147/gamma*
_output_shapes	
:*
dtype0

batch_normalization_147/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_147/beta

0batch_normalization_147/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_147/beta*
_output_shapes	
:*
dtype0

#batch_normalization_147/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_147/moving_mean

7batch_normalization_147/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_147/moving_mean*
_output_shapes	
:*
dtype0
§
'batch_normalization_147/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_147/moving_variance
 
;batch_normalization_147/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_147/moving_variance*
_output_shapes	
:*
dtype0

Dense_layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*$
shared_nameDense_layer1/kernel
|
'Dense_layer1/kernel/Read/ReadVariableOpReadVariableOpDense_layer1/kernel*
_output_shapes
:	2*
dtype0
z
Dense_layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*"
shared_nameDense_layer1/bias
s
%Dense_layer1/bias/Read/ReadVariableOpReadVariableOpDense_layer1/bias*
_output_shapes
:2*
dtype0

Dense_layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2d*$
shared_nameDense_layer2/kernel
{
'Dense_layer2/kernel/Read/ReadVariableOpReadVariableOpDense_layer2/kernel*
_output_shapes

:2d*
dtype0
z
Dense_layer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*"
shared_nameDense_layer2/bias
s
%Dense_layer2/bias/Read/ReadVariableOpReadVariableOpDense_layer2/bias*
_output_shapes
:d*
dtype0

output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*$
shared_nameoutput_layer/kernel
{
'output_layer/kernel/Read/ReadVariableOpReadVariableOpoutput_layer/kernel*
_output_shapes

:d*
dtype0
z
output_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameoutput_layer/bias
s
%output_layer/bias/Read/ReadVariableOpReadVariableOpoutput_layer/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
 
$Adam/batch_normalization_143/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_143/gamma/m

8Adam/batch_normalization_143/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_143/gamma/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_143/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_143/beta/m

7Adam/batch_normalization_143/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_143/beta/m*
_output_shapes
:*
dtype0

Adam/conv_layer1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/conv_layer1/kernel/m

-Adam/conv_layer1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_layer1/kernel/m*&
_output_shapes
: *
dtype0

Adam/conv_layer1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv_layer1/bias/m

+Adam/conv_layer1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_layer1/bias/m*
_output_shapes
: *
dtype0
 
$Adam/batch_normalization_144/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adam/batch_normalization_144/gamma/m

8Adam/batch_normalization_144/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_144/gamma/m*
_output_shapes
: *
dtype0

#Adam/batch_normalization_144/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_144/beta/m

7Adam/batch_normalization_144/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_144/beta/m*
_output_shapes
: *
dtype0

Adam/conv_layer2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameAdam/conv_layer2/kernel/m

-Adam/conv_layer2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_layer2/kernel/m*&
_output_shapes
: @*
dtype0

Adam/conv_layer2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv_layer2/bias/m

+Adam/conv_layer2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_layer2/bias/m*
_output_shapes
:@*
dtype0
 
$Adam/batch_normalization_145/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$Adam/batch_normalization_145/gamma/m

8Adam/batch_normalization_145/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_145/gamma/m*
_output_shapes
:@*
dtype0

#Adam/batch_normalization_145/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_145/beta/m

7Adam/batch_normalization_145/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_145/beta/m*
_output_shapes
:@*
dtype0

Adam/conv_layer3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameAdam/conv_layer3/kernel/m

-Adam/conv_layer3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_layer3/kernel/m*'
_output_shapes
:@*
dtype0

Adam/conv_layer3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv_layer3/bias/m

+Adam/conv_layer3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_layer3/bias/m*
_output_shapes	
:*
dtype0
¡
$Adam/batch_normalization_146/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_146/gamma/m

8Adam/batch_normalization_146/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_146/gamma/m*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_146/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_146/beta/m

7Adam/batch_normalization_146/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_146/beta/m*
_output_shapes	
:*
dtype0

Adam/conv_layer4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/conv_layer4/kernel/m

-Adam/conv_layer4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_layer4/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv_layer4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv_layer4/bias/m

+Adam/conv_layer4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_layer4/bias/m*
_output_shapes	
:*
dtype0
¡
$Adam/batch_normalization_147/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_147/gamma/m

8Adam/batch_normalization_147/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_147/gamma/m*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_147/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_147/beta/m

7Adam/batch_normalization_147/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_147/beta/m*
_output_shapes	
:*
dtype0

Adam/Dense_layer1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*+
shared_nameAdam/Dense_layer1/kernel/m

.Adam/Dense_layer1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_layer1/kernel/m*
_output_shapes
:	2*
dtype0

Adam/Dense_layer1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*)
shared_nameAdam/Dense_layer1/bias/m

,Adam/Dense_layer1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_layer1/bias/m*
_output_shapes
:2*
dtype0

Adam/Dense_layer2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2d*+
shared_nameAdam/Dense_layer2/kernel/m

.Adam/Dense_layer2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_layer2/kernel/m*
_output_shapes

:2d*
dtype0

Adam/Dense_layer2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*)
shared_nameAdam/Dense_layer2/bias/m

,Adam/Dense_layer2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_layer2/bias/m*
_output_shapes
:d*
dtype0

Adam/output_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*+
shared_nameAdam/output_layer/kernel/m

.Adam/output_layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_layer/kernel/m*
_output_shapes

:d*
dtype0

Adam/output_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/output_layer/bias/m

,Adam/output_layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/output_layer/bias/m*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_143/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_143/gamma/v

8Adam/batch_normalization_143/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_143/gamma/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_143/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_143/beta/v

7Adam/batch_normalization_143/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_143/beta/v*
_output_shapes
:*
dtype0

Adam/conv_layer1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/conv_layer1/kernel/v

-Adam/conv_layer1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_layer1/kernel/v*&
_output_shapes
: *
dtype0

Adam/conv_layer1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv_layer1/bias/v

+Adam/conv_layer1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_layer1/bias/v*
_output_shapes
: *
dtype0
 
$Adam/batch_normalization_144/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adam/batch_normalization_144/gamma/v

8Adam/batch_normalization_144/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_144/gamma/v*
_output_shapes
: *
dtype0

#Adam/batch_normalization_144/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_144/beta/v

7Adam/batch_normalization_144/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_144/beta/v*
_output_shapes
: *
dtype0

Adam/conv_layer2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameAdam/conv_layer2/kernel/v

-Adam/conv_layer2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_layer2/kernel/v*&
_output_shapes
: @*
dtype0

Adam/conv_layer2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv_layer2/bias/v

+Adam/conv_layer2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_layer2/bias/v*
_output_shapes
:@*
dtype0
 
$Adam/batch_normalization_145/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$Adam/batch_normalization_145/gamma/v

8Adam/batch_normalization_145/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_145/gamma/v*
_output_shapes
:@*
dtype0

#Adam/batch_normalization_145/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_145/beta/v

7Adam/batch_normalization_145/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_145/beta/v*
_output_shapes
:@*
dtype0

Adam/conv_layer3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameAdam/conv_layer3/kernel/v

-Adam/conv_layer3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_layer3/kernel/v*'
_output_shapes
:@*
dtype0

Adam/conv_layer3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv_layer3/bias/v

+Adam/conv_layer3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_layer3/bias/v*
_output_shapes	
:*
dtype0
¡
$Adam/batch_normalization_146/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_146/gamma/v

8Adam/batch_normalization_146/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_146/gamma/v*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_146/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_146/beta/v

7Adam/batch_normalization_146/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_146/beta/v*
_output_shapes	
:*
dtype0

Adam/conv_layer4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/conv_layer4/kernel/v

-Adam/conv_layer4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_layer4/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv_layer4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv_layer4/bias/v

+Adam/conv_layer4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_layer4/bias/v*
_output_shapes	
:*
dtype0
¡
$Adam/batch_normalization_147/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_147/gamma/v

8Adam/batch_normalization_147/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_147/gamma/v*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_147/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_147/beta/v

7Adam/batch_normalization_147/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_147/beta/v*
_output_shapes	
:*
dtype0

Adam/Dense_layer1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*+
shared_nameAdam/Dense_layer1/kernel/v

.Adam/Dense_layer1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_layer1/kernel/v*
_output_shapes
:	2*
dtype0

Adam/Dense_layer1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*)
shared_nameAdam/Dense_layer1/bias/v

,Adam/Dense_layer1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_layer1/bias/v*
_output_shapes
:2*
dtype0

Adam/Dense_layer2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2d*+
shared_nameAdam/Dense_layer2/kernel/v

.Adam/Dense_layer2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_layer2/kernel/v*
_output_shapes

:2d*
dtype0

Adam/Dense_layer2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*)
shared_nameAdam/Dense_layer2/bias/v

,Adam/Dense_layer2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_layer2/bias/v*
_output_shapes
:d*
dtype0

Adam/output_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*+
shared_nameAdam/output_layer/kernel/v

.Adam/output_layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_layer/kernel/v*
_output_shapes

:d*
dtype0

Adam/output_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/output_layer/bias/v

,Adam/output_layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/output_layer/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ã
value¸B´ B¬

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer-13
layer_with_weights-9
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures

axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
 	keras_api
h

!kernel
"bias
#regularization_losses
$trainable_variables
%	variables
&	keras_api
R
'regularization_losses
(trainable_variables
)	variables
*	keras_api

+axis
	,gamma
-beta
.moving_mean
/moving_variance
0regularization_losses
1trainable_variables
2	variables
3	keras_api
h

4kernel
5bias
6regularization_losses
7trainable_variables
8	variables
9	keras_api
R
:regularization_losses
;trainable_variables
<	variables
=	keras_api

>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
h

Gkernel
Hbias
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
R
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api

Qaxis
	Rgamma
Sbeta
Tmoving_mean
Umoving_variance
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
h

Zkernel
[bias
\regularization_losses
]trainable_variables
^	variables
_	keras_api
R
`regularization_losses
atrainable_variables
b	variables
c	keras_api

daxis
	egamma
fbeta
gmoving_mean
hmoving_variance
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
R
mregularization_losses
ntrainable_variables
o	variables
p	keras_api
h

qkernel
rbias
sregularization_losses
ttrainable_variables
u	variables
v	keras_api
h

wkernel
xbias
yregularization_losses
ztrainable_variables
{	variables
|	keras_api
k

}kernel
~bias
regularization_losses
trainable_variables
	variables
	keras_api
¥
	iter
beta_1
beta_2

decay
learning_ratemímî!mï"mð,mñ-mò4mó5mô?mõ@möGm÷HmøRmùSmúZmû[müemýfmþqmÿrmwmxm}m~mvv!v"v,v-v4v5v?v@vGvHvRvSvZv[vevfvqvrvwvxv}v~v
 
¶
0
1
!2
"3
,4
-5
46
57
?8
@9
G10
H11
R12
S13
Z14
[15
e16
f17
q18
r19
w20
x21
}22
~23

0
1
2
3
!4
"5
,6
-7
.8
/9
410
511
?12
@13
A14
B15
G16
H17
R18
S19
T20
U21
Z22
[23
e24
f25
g26
h27
q28
r29
w30
x31
}32
~33
²
non_trainable_variables
metrics
regularization_losses
layer_metrics
layers
 layer_regularization_losses
trainable_variables
	variables
 
 
hf
VARIABLE_VALUEbatch_normalization_143/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_143/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_143/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_143/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
2
3
²
non_trainable_variables
metrics
regularization_losses
layer_metrics
layers
 layer_regularization_losses
trainable_variables
	variables
^\
VARIABLE_VALUEconv_layer1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv_layer1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

!0
"1

!0
"1
²
non_trainable_variables
metrics
#regularization_losses
layer_metrics
layers
 layer_regularization_losses
$trainable_variables
%	variables
 
 
 
²
non_trainable_variables
metrics
'regularization_losses
layer_metrics
layers
 layer_regularization_losses
(trainable_variables
)	variables
 
hf
VARIABLE_VALUEbatch_normalization_144/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_144/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_144/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_144/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

,0
-1

,0
-1
.2
/3
²
non_trainable_variables
metrics
0regularization_losses
layer_metrics
layers
  layer_regularization_losses
1trainable_variables
2	variables
^\
VARIABLE_VALUEconv_layer2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv_layer2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51

40
51
²
¡non_trainable_variables
¢metrics
6regularization_losses
£layer_metrics
¤layers
 ¥layer_regularization_losses
7trainable_variables
8	variables
 
 
 
²
¦non_trainable_variables
§metrics
:regularization_losses
¨layer_metrics
©layers
 ªlayer_regularization_losses
;trainable_variables
<	variables
 
hf
VARIABLE_VALUEbatch_normalization_145/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_145/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_145/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_145/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

?0
@1

?0
@1
A2
B3
²
«non_trainable_variables
¬metrics
Cregularization_losses
­layer_metrics
®layers
 ¯layer_regularization_losses
Dtrainable_variables
E	variables
^\
VARIABLE_VALUEconv_layer3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv_layer3/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

G0
H1

G0
H1
²
°non_trainable_variables
±metrics
Iregularization_losses
²layer_metrics
³layers
 ´layer_regularization_losses
Jtrainable_variables
K	variables
 
 
 
²
µnon_trainable_variables
¶metrics
Mregularization_losses
·layer_metrics
¸layers
 ¹layer_regularization_losses
Ntrainable_variables
O	variables
 
hf
VARIABLE_VALUEbatch_normalization_146/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_146/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_146/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_146/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

R0
S1

R0
S1
T2
U3
²
ºnon_trainable_variables
»metrics
Vregularization_losses
¼layer_metrics
½layers
 ¾layer_regularization_losses
Wtrainable_variables
X	variables
^\
VARIABLE_VALUEconv_layer4/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv_layer4/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Z0
[1

Z0
[1
²
¿non_trainable_variables
Àmetrics
\regularization_losses
Álayer_metrics
Âlayers
 Ãlayer_regularization_losses
]trainable_variables
^	variables
 
 
 
²
Änon_trainable_variables
Åmetrics
`regularization_losses
Ælayer_metrics
Çlayers
 Èlayer_regularization_losses
atrainable_variables
b	variables
 
hf
VARIABLE_VALUEbatch_normalization_147/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_147/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_147/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_147/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

e0
f1

e0
f1
g2
h3
²
Énon_trainable_variables
Êmetrics
iregularization_losses
Ëlayer_metrics
Ìlayers
 Ílayer_regularization_losses
jtrainable_variables
k	variables
 
 
 
²
Înon_trainable_variables
Ïmetrics
mregularization_losses
Ðlayer_metrics
Ñlayers
 Òlayer_regularization_losses
ntrainable_variables
o	variables
_]
VARIABLE_VALUEDense_layer1/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEDense_layer1/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

q0
r1

q0
r1
²
Ónon_trainable_variables
Ômetrics
sregularization_losses
Õlayer_metrics
Ölayers
 ×layer_regularization_losses
ttrainable_variables
u	variables
`^
VARIABLE_VALUEDense_layer2/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEDense_layer2/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

w0
x1

w0
x1
²
Ønon_trainable_variables
Ùmetrics
yregularization_losses
Úlayer_metrics
Ûlayers
 Ülayer_regularization_losses
ztrainable_variables
{	variables
`^
VARIABLE_VALUEoutput_layer/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEoutput_layer/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 

}0
~1

}0
~1
´
Ýnon_trainable_variables
Þmetrics
regularization_losses
ßlayer_metrics
àlayers
 álayer_regularization_losses
trainable_variables
	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
F
0
1
.2
/3
A4
B5
T6
U7
g8
h9

â0
ã1
 
~
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

.0
/1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

A0
B1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

T0
U1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

g0
h1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

ätotal

åcount
æ	variables
ç	keras_api
I

ètotal

écount
ê
_fn_kwargs
ë	variables
ì	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

ä0
å1

æ	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

è0
é1

ë	variables

VARIABLE_VALUE$Adam/batch_normalization_143/gamma/mQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_143/beta/mPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv_layer1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv_layer1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/batch_normalization_144/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_144/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv_layer2/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv_layer2/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/batch_normalization_145/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_145/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv_layer3/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv_layer3/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/batch_normalization_146/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_146/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv_layer4/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv_layer4/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/batch_normalization_147/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_147/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/Dense_layer1/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Dense_layer1/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/Dense_layer2/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/Dense_layer2/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/output_layer/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/output_layer/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/batch_normalization_143/gamma/vQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_143/beta/vPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv_layer1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv_layer1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/batch_normalization_144/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_144/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv_layer2/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv_layer2/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/batch_normalization_145/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_145/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv_layer3/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv_layer3/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/batch_normalization_146/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_146/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv_layer4/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv_layer4/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/batch_normalization_147/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_147/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/Dense_layer1/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Dense_layer1/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/Dense_layer2/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/Dense_layer2/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/output_layer/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/output_layer/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_layerPlaceholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ¬¬
Ä

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerbatch_normalization_143/gammabatch_normalization_143/beta#batch_normalization_143/moving_mean'batch_normalization_143/moving_varianceconv_layer1/kernelconv_layer1/biasbatch_normalization_144/gammabatch_normalization_144/beta#batch_normalization_144/moving_mean'batch_normalization_144/moving_varianceconv_layer2/kernelconv_layer2/biasbatch_normalization_145/gammabatch_normalization_145/beta#batch_normalization_145/moving_mean'batch_normalization_145/moving_varianceconv_layer3/kernelconv_layer3/biasbatch_normalization_146/gammabatch_normalization_146/beta#batch_normalization_146/moving_mean'batch_normalization_146/moving_varianceconv_layer4/kernelconv_layer4/biasbatch_normalization_147/gammabatch_normalization_147/beta#batch_normalization_147/moving_mean'batch_normalization_147/moving_varianceDense_layer1/kernelDense_layer1/biasDense_layer2/kernelDense_layer2/biasoutput_layer/kerneloutput_layer/bias*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_211045
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
%
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename1batch_normalization_143/gamma/Read/ReadVariableOp0batch_normalization_143/beta/Read/ReadVariableOp7batch_normalization_143/moving_mean/Read/ReadVariableOp;batch_normalization_143/moving_variance/Read/ReadVariableOp&conv_layer1/kernel/Read/ReadVariableOp$conv_layer1/bias/Read/ReadVariableOp1batch_normalization_144/gamma/Read/ReadVariableOp0batch_normalization_144/beta/Read/ReadVariableOp7batch_normalization_144/moving_mean/Read/ReadVariableOp;batch_normalization_144/moving_variance/Read/ReadVariableOp&conv_layer2/kernel/Read/ReadVariableOp$conv_layer2/bias/Read/ReadVariableOp1batch_normalization_145/gamma/Read/ReadVariableOp0batch_normalization_145/beta/Read/ReadVariableOp7batch_normalization_145/moving_mean/Read/ReadVariableOp;batch_normalization_145/moving_variance/Read/ReadVariableOp&conv_layer3/kernel/Read/ReadVariableOp$conv_layer3/bias/Read/ReadVariableOp1batch_normalization_146/gamma/Read/ReadVariableOp0batch_normalization_146/beta/Read/ReadVariableOp7batch_normalization_146/moving_mean/Read/ReadVariableOp;batch_normalization_146/moving_variance/Read/ReadVariableOp&conv_layer4/kernel/Read/ReadVariableOp$conv_layer4/bias/Read/ReadVariableOp1batch_normalization_147/gamma/Read/ReadVariableOp0batch_normalization_147/beta/Read/ReadVariableOp7batch_normalization_147/moving_mean/Read/ReadVariableOp;batch_normalization_147/moving_variance/Read/ReadVariableOp'Dense_layer1/kernel/Read/ReadVariableOp%Dense_layer1/bias/Read/ReadVariableOp'Dense_layer2/kernel/Read/ReadVariableOp%Dense_layer2/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp8Adam/batch_normalization_143/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_143/beta/m/Read/ReadVariableOp-Adam/conv_layer1/kernel/m/Read/ReadVariableOp+Adam/conv_layer1/bias/m/Read/ReadVariableOp8Adam/batch_normalization_144/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_144/beta/m/Read/ReadVariableOp-Adam/conv_layer2/kernel/m/Read/ReadVariableOp+Adam/conv_layer2/bias/m/Read/ReadVariableOp8Adam/batch_normalization_145/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_145/beta/m/Read/ReadVariableOp-Adam/conv_layer3/kernel/m/Read/ReadVariableOp+Adam/conv_layer3/bias/m/Read/ReadVariableOp8Adam/batch_normalization_146/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_146/beta/m/Read/ReadVariableOp-Adam/conv_layer4/kernel/m/Read/ReadVariableOp+Adam/conv_layer4/bias/m/Read/ReadVariableOp8Adam/batch_normalization_147/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_147/beta/m/Read/ReadVariableOp.Adam/Dense_layer1/kernel/m/Read/ReadVariableOp,Adam/Dense_layer1/bias/m/Read/ReadVariableOp.Adam/Dense_layer2/kernel/m/Read/ReadVariableOp,Adam/Dense_layer2/bias/m/Read/ReadVariableOp.Adam/output_layer/kernel/m/Read/ReadVariableOp,Adam/output_layer/bias/m/Read/ReadVariableOp8Adam/batch_normalization_143/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_143/beta/v/Read/ReadVariableOp-Adam/conv_layer1/kernel/v/Read/ReadVariableOp+Adam/conv_layer1/bias/v/Read/ReadVariableOp8Adam/batch_normalization_144/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_144/beta/v/Read/ReadVariableOp-Adam/conv_layer2/kernel/v/Read/ReadVariableOp+Adam/conv_layer2/bias/v/Read/ReadVariableOp8Adam/batch_normalization_145/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_145/beta/v/Read/ReadVariableOp-Adam/conv_layer3/kernel/v/Read/ReadVariableOp+Adam/conv_layer3/bias/v/Read/ReadVariableOp8Adam/batch_normalization_146/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_146/beta/v/Read/ReadVariableOp-Adam/conv_layer4/kernel/v/Read/ReadVariableOp+Adam/conv_layer4/bias/v/Read/ReadVariableOp8Adam/batch_normalization_147/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_147/beta/v/Read/ReadVariableOp.Adam/Dense_layer1/kernel/v/Read/ReadVariableOp,Adam/Dense_layer1/bias/v/Read/ReadVariableOp.Adam/Dense_layer2/kernel/v/Read/ReadVariableOp,Adam/Dense_layer2/bias/v/Read/ReadVariableOp.Adam/output_layer/kernel/v/Read/ReadVariableOp,Adam/output_layer/bias/v/Read/ReadVariableOpConst*h
Tina
_2]	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_212535
ò
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization_143/gammabatch_normalization_143/beta#batch_normalization_143/moving_mean'batch_normalization_143/moving_varianceconv_layer1/kernelconv_layer1/biasbatch_normalization_144/gammabatch_normalization_144/beta#batch_normalization_144/moving_mean'batch_normalization_144/moving_varianceconv_layer2/kernelconv_layer2/biasbatch_normalization_145/gammabatch_normalization_145/beta#batch_normalization_145/moving_mean'batch_normalization_145/moving_varianceconv_layer3/kernelconv_layer3/biasbatch_normalization_146/gammabatch_normalization_146/beta#batch_normalization_146/moving_mean'batch_normalization_146/moving_varianceconv_layer4/kernelconv_layer4/biasbatch_normalization_147/gammabatch_normalization_147/beta#batch_normalization_147/moving_mean'batch_normalization_147/moving_varianceDense_layer1/kernelDense_layer1/biasDense_layer2/kernelDense_layer2/biasoutput_layer/kerneloutput_layer/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1$Adam/batch_normalization_143/gamma/m#Adam/batch_normalization_143/beta/mAdam/conv_layer1/kernel/mAdam/conv_layer1/bias/m$Adam/batch_normalization_144/gamma/m#Adam/batch_normalization_144/beta/mAdam/conv_layer2/kernel/mAdam/conv_layer2/bias/m$Adam/batch_normalization_145/gamma/m#Adam/batch_normalization_145/beta/mAdam/conv_layer3/kernel/mAdam/conv_layer3/bias/m$Adam/batch_normalization_146/gamma/m#Adam/batch_normalization_146/beta/mAdam/conv_layer4/kernel/mAdam/conv_layer4/bias/m$Adam/batch_normalization_147/gamma/m#Adam/batch_normalization_147/beta/mAdam/Dense_layer1/kernel/mAdam/Dense_layer1/bias/mAdam/Dense_layer2/kernel/mAdam/Dense_layer2/bias/mAdam/output_layer/kernel/mAdam/output_layer/bias/m$Adam/batch_normalization_143/gamma/v#Adam/batch_normalization_143/beta/vAdam/conv_layer1/kernel/vAdam/conv_layer1/bias/v$Adam/batch_normalization_144/gamma/v#Adam/batch_normalization_144/beta/vAdam/conv_layer2/kernel/vAdam/conv_layer2/bias/v$Adam/batch_normalization_145/gamma/v#Adam/batch_normalization_145/beta/vAdam/conv_layer3/kernel/vAdam/conv_layer3/bias/v$Adam/batch_normalization_146/gamma/v#Adam/batch_normalization_146/beta/vAdam/conv_layer4/kernel/vAdam/conv_layer4/bias/v$Adam/batch_normalization_147/gamma/v#Adam/batch_normalization_147/beta/vAdam/Dense_layer1/kernel/vAdam/Dense_layer1/bias/vAdam/Dense_layer2/kernel/vAdam/Dense_layer2/bias/vAdam/output_layer/kernel/vAdam/output_layer/bias/v*g
Tin`
^2\*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_212818ú±
	
¯
G__inference_conv_layer3_layer_call_and_return_conditional_losses_211894

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿGG*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿGG2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿGG2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿGG2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿII@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@
 
_user_specified_nameinputs
Þ
«
8__inference_batch_normalization_145_layer_call_fn_211883

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_145_layer_call_and_return_conditional_losses_2102272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿII@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@
 
_user_specified_nameinputs


S__inference_batch_normalization_143_layer_call_and_return_conditional_losses_209496

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
°
S__inference_batch_normalization_145_layer_call_and_return_conditional_losses_211775

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
´
N
2__inference_max_pooling2d_130_layer_call_fn_209635

inputs
identityñ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_130_layer_call_and_return_conditional_losses_2096292
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

X
<__inference_global_average_pooling2d_26_layer_call_fn_209984

inputs
identityá
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling2d_26_layer_call_and_return_conditional_losses_2099782
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð^

I__inference_sequential_24_layer_call_and_return_conditional_losses_210637
input_layer"
batch_normalization_143_210551"
batch_normalization_143_210553"
batch_normalization_143_210555"
batch_normalization_143_210557
conv_layer1_210560
conv_layer1_210562"
batch_normalization_144_210566"
batch_normalization_144_210568"
batch_normalization_144_210570"
batch_normalization_144_210572
conv_layer2_210575
conv_layer2_210577"
batch_normalization_145_210581"
batch_normalization_145_210583"
batch_normalization_145_210585"
batch_normalization_145_210587
conv_layer3_210590
conv_layer3_210592"
batch_normalization_146_210596"
batch_normalization_146_210598"
batch_normalization_146_210600"
batch_normalization_146_210602
conv_layer4_210605
conv_layer4_210607"
batch_normalization_147_210611"
batch_normalization_147_210613"
batch_normalization_147_210615"
batch_normalization_147_210617
dense_layer1_210621
dense_layer1_210623
dense_layer2_210626
dense_layer2_210628
output_layer_210631
output_layer_210633
identity¢$Dense_layer1/StatefulPartitionedCall¢$Dense_layer2/StatefulPartitionedCall¢/batch_normalization_143/StatefulPartitionedCall¢/batch_normalization_144/StatefulPartitionedCall¢/batch_normalization_145/StatefulPartitionedCall¢/batch_normalization_146/StatefulPartitionedCall¢/batch_normalization_147/StatefulPartitionedCall¢#conv_layer1/StatefulPartitionedCall¢#conv_layer2/StatefulPartitionedCall¢#conv_layer3/StatefulPartitionedCall¢#conv_layer4/StatefulPartitionedCall¢$output_layer/StatefulPartitionedCallµ
/batch_normalization_143/StatefulPartitionedCallStatefulPartitionedCallinput_layerbatch_normalization_143_210551batch_normalization_143_210553batch_normalization_143_210555batch_normalization_143_210557*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_143_layer_call_and_return_conditional_losses_21002521
/batch_normalization_143/StatefulPartitionedCallâ
#conv_layer1/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_143/StatefulPartitionedCall:output:0conv_layer1_210560conv_layer1_210562*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©© *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_layer1_layer_call_and_return_conditional_losses_2100722%
#conv_layer1/StatefulPartitionedCall¢
!max_pooling2d_129/PartitionedCallPartitionedCall,conv_layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_129_layer_call_and_return_conditional_losses_2095132#
!max_pooling2d_129/PartitionedCallÔ
/batch_normalization_144/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_129/PartitionedCall:output:0batch_normalization_144_210566batch_normalization_144_210568batch_normalization_144_210570batch_normalization_144_210572*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_144_layer_call_and_return_conditional_losses_21012621
/batch_normalization_144/StatefulPartitionedCallâ
#conv_layer2/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_144/StatefulPartitionedCall:output:0conv_layer2_210575conv_layer2_210577*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_layer2_layer_call_and_return_conditional_losses_2101732%
#conv_layer2/StatefulPartitionedCall 
!max_pooling2d_130/PartitionedCallPartitionedCall,conv_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_130_layer_call_and_return_conditional_losses_2096292#
!max_pooling2d_130/PartitionedCallÒ
/batch_normalization_145/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_130/PartitionedCall:output:0batch_normalization_145_210581batch_normalization_145_210583batch_normalization_145_210585batch_normalization_145_210587*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_145_layer_call_and_return_conditional_losses_21022721
/batch_normalization_145/StatefulPartitionedCallá
#conv_layer3/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_145/StatefulPartitionedCall:output:0conv_layer3_210590conv_layer3_210592*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿGG*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_layer3_layer_call_and_return_conditional_losses_2102742%
#conv_layer3/StatefulPartitionedCall¡
!max_pooling2d_131/PartitionedCallPartitionedCall,conv_layer3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_131_layer_call_and_return_conditional_losses_2097452#
!max_pooling2d_131/PartitionedCallÓ
/batch_normalization_146/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_131/PartitionedCall:output:0batch_normalization_146_210596batch_normalization_146_210598batch_normalization_146_210600batch_normalization_146_210602*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_146_layer_call_and_return_conditional_losses_21032821
/batch_normalization_146/StatefulPartitionedCallá
#conv_layer4/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_146/StatefulPartitionedCall:output:0conv_layer4_210605conv_layer4_210607*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_layer4_layer_call_and_return_conditional_losses_2103752%
#conv_layer4/StatefulPartitionedCall¡
!max_pooling2d_132/PartitionedCallPartitionedCall,conv_layer4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_132_layer_call_and_return_conditional_losses_2098612#
!max_pooling2d_132/PartitionedCallÓ
/batch_normalization_147/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_132/PartitionedCall:output:0batch_normalization_147_210611batch_normalization_147_210613batch_normalization_147_210615batch_normalization_147_210617*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_147_layer_call_and_return_conditional_losses_21042921
/batch_normalization_147/StatefulPartitionedCallÃ
+global_average_pooling2d_26/PartitionedCallPartitionedCall8batch_normalization_147/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling2d_26_layer_call_and_return_conditional_losses_2099782-
+global_average_pooling2d_26/PartitionedCallÙ
$Dense_layer1/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling2d_26/PartitionedCall:output:0dense_layer1_210621dense_layer1_210623*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Dense_layer1_layer_call_and_return_conditional_losses_2104772&
$Dense_layer1/StatefulPartitionedCallÒ
$Dense_layer2/StatefulPartitionedCallStatefulPartitionedCall-Dense_layer1/StatefulPartitionedCall:output:0dense_layer2_210626dense_layer2_210628*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Dense_layer2_layer_call_and_return_conditional_losses_2105042&
$Dense_layer2/StatefulPartitionedCallÒ
$output_layer/StatefulPartitionedCallStatefulPartitionedCall-Dense_layer2/StatefulPartitionedCall:output:0output_layer_210631output_layer_210633*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_2105312&
$output_layer/StatefulPartitionedCall
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0%^Dense_layer1/StatefulPartitionedCall%^Dense_layer2/StatefulPartitionedCall0^batch_normalization_143/StatefulPartitionedCall0^batch_normalization_144/StatefulPartitionedCall0^batch_normalization_145/StatefulPartitionedCall0^batch_normalization_146/StatefulPartitionedCall0^batch_normalization_147/StatefulPartitionedCall$^conv_layer1/StatefulPartitionedCall$^conv_layer2/StatefulPartitionedCall$^conv_layer3/StatefulPartitionedCall$^conv_layer4/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*º
_input_shapes¨
¥:ÿÿÿÿÿÿÿÿÿ¬¬::::::::::::::::::::::::::::::::::2L
$Dense_layer1/StatefulPartitionedCall$Dense_layer1/StatefulPartitionedCall2L
$Dense_layer2/StatefulPartitionedCall$Dense_layer2/StatefulPartitionedCall2b
/batch_normalization_143/StatefulPartitionedCall/batch_normalization_143/StatefulPartitionedCall2b
/batch_normalization_144/StatefulPartitionedCall/batch_normalization_144/StatefulPartitionedCall2b
/batch_normalization_145/StatefulPartitionedCall/batch_normalization_145/StatefulPartitionedCall2b
/batch_normalization_146/StatefulPartitionedCall/batch_normalization_146/StatefulPartitionedCall2b
/batch_normalization_147/StatefulPartitionedCall/batch_normalization_147/StatefulPartitionedCall2J
#conv_layer1/StatefulPartitionedCall#conv_layer1/StatefulPartitionedCall2J
#conv_layer2/StatefulPartitionedCall#conv_layer2/StatefulPartitionedCall2J
#conv_layer3/StatefulPartitionedCall#conv_layer3/StatefulPartitionedCall2J
#conv_layer4/StatefulPartitionedCall#conv_layer4/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
%
_user_specified_nameinput_layer

°
S__inference_batch_normalization_144_layer_call_and_return_conditional_losses_210108

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ú
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_129_layer_call_and_return_conditional_losses_209513

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä
«
8__inference_batch_normalization_143_layer_call_fn_211574

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_143_layer_call_and_return_conditional_losses_2100072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ¬¬::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
 
_user_specified_nameinputs
Ú

S__inference_batch_normalization_143_layer_call_and_return_conditional_losses_211561

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ¬¬:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3r
IdentityIdentityFusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ¬¬:::::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
 
_user_specified_nameinputs


S__inference_batch_normalization_145_layer_call_and_return_conditional_losses_211793

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¦
«
8__inference_batch_normalization_144_layer_call_fn_211671

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallº
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_144_layer_call_and_return_conditional_losses_2096122
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ª
«
8__inference_batch_normalization_146_layer_call_fn_212031

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_146_layer_call_and_return_conditional_losses_2098442
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ¡
°
!__inference__wrapped_model_209403
input_layerA
=sequential_24_batch_normalization_143_readvariableop_resourceC
?sequential_24_batch_normalization_143_readvariableop_1_resourceR
Nsequential_24_batch_normalization_143_fusedbatchnormv3_readvariableop_resourceT
Psequential_24_batch_normalization_143_fusedbatchnormv3_readvariableop_1_resource<
8sequential_24_conv_layer1_conv2d_readvariableop_resource=
9sequential_24_conv_layer1_biasadd_readvariableop_resourceA
=sequential_24_batch_normalization_144_readvariableop_resourceC
?sequential_24_batch_normalization_144_readvariableop_1_resourceR
Nsequential_24_batch_normalization_144_fusedbatchnormv3_readvariableop_resourceT
Psequential_24_batch_normalization_144_fusedbatchnormv3_readvariableop_1_resource<
8sequential_24_conv_layer2_conv2d_readvariableop_resource=
9sequential_24_conv_layer2_biasadd_readvariableop_resourceA
=sequential_24_batch_normalization_145_readvariableop_resourceC
?sequential_24_batch_normalization_145_readvariableop_1_resourceR
Nsequential_24_batch_normalization_145_fusedbatchnormv3_readvariableop_resourceT
Psequential_24_batch_normalization_145_fusedbatchnormv3_readvariableop_1_resource<
8sequential_24_conv_layer3_conv2d_readvariableop_resource=
9sequential_24_conv_layer3_biasadd_readvariableop_resourceA
=sequential_24_batch_normalization_146_readvariableop_resourceC
?sequential_24_batch_normalization_146_readvariableop_1_resourceR
Nsequential_24_batch_normalization_146_fusedbatchnormv3_readvariableop_resourceT
Psequential_24_batch_normalization_146_fusedbatchnormv3_readvariableop_1_resource<
8sequential_24_conv_layer4_conv2d_readvariableop_resource=
9sequential_24_conv_layer4_biasadd_readvariableop_resourceA
=sequential_24_batch_normalization_147_readvariableop_resourceC
?sequential_24_batch_normalization_147_readvariableop_1_resourceR
Nsequential_24_batch_normalization_147_fusedbatchnormv3_readvariableop_resourceT
Psequential_24_batch_normalization_147_fusedbatchnormv3_readvariableop_1_resource=
9sequential_24_dense_layer1_matmul_readvariableop_resource>
:sequential_24_dense_layer1_biasadd_readvariableop_resource=
9sequential_24_dense_layer2_matmul_readvariableop_resource>
:sequential_24_dense_layer2_biasadd_readvariableop_resource=
9sequential_24_output_layer_matmul_readvariableop_resource>
:sequential_24_output_layer_biasadd_readvariableop_resource
identityæ
4sequential_24/batch_normalization_143/ReadVariableOpReadVariableOp=sequential_24_batch_normalization_143_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential_24/batch_normalization_143/ReadVariableOpì
6sequential_24/batch_normalization_143/ReadVariableOp_1ReadVariableOp?sequential_24_batch_normalization_143_readvariableop_1_resource*
_output_shapes
:*
dtype028
6sequential_24/batch_normalization_143/ReadVariableOp_1
Esequential_24/batch_normalization_143/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_24_batch_normalization_143_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02G
Esequential_24/batch_normalization_143/FusedBatchNormV3/ReadVariableOp
Gsequential_24/batch_normalization_143/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_24_batch_normalization_143_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02I
Gsequential_24/batch_normalization_143/FusedBatchNormV3/ReadVariableOp_1µ
6sequential_24/batch_normalization_143/FusedBatchNormV3FusedBatchNormV3input_layer<sequential_24/batch_normalization_143/ReadVariableOp:value:0>sequential_24/batch_normalization_143/ReadVariableOp_1:value:0Msequential_24/batch_normalization_143/FusedBatchNormV3/ReadVariableOp:value:0Osequential_24/batch_normalization_143/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ¬¬:::::*
epsilon%o:*
is_training( 28
6sequential_24/batch_normalization_143/FusedBatchNormV3ã
/sequential_24/conv_layer1/Conv2D/ReadVariableOpReadVariableOp8sequential_24_conv_layer1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype021
/sequential_24/conv_layer1/Conv2D/ReadVariableOp¨
 sequential_24/conv_layer1/Conv2DConv2D:sequential_24/batch_normalization_143/FusedBatchNormV3:y:07sequential_24/conv_layer1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©© *
paddingVALID*
strides
2"
 sequential_24/conv_layer1/Conv2DÚ
0sequential_24/conv_layer1/BiasAdd/ReadVariableOpReadVariableOp9sequential_24_conv_layer1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0sequential_24/conv_layer1/BiasAdd/ReadVariableOpò
!sequential_24/conv_layer1/BiasAddBiasAdd)sequential_24/conv_layer1/Conv2D:output:08sequential_24/conv_layer1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©© 2#
!sequential_24/conv_layer1/BiasAdd°
sequential_24/conv_layer1/ReluRelu*sequential_24/conv_layer1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©© 2 
sequential_24/conv_layer1/Reluú
'sequential_24/max_pooling2d_129/MaxPoolMaxPool,sequential_24/conv_layer1/Relu:activations:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2)
'sequential_24/max_pooling2d_129/MaxPoolæ
4sequential_24/batch_normalization_144/ReadVariableOpReadVariableOp=sequential_24_batch_normalization_144_readvariableop_resource*
_output_shapes
: *
dtype026
4sequential_24/batch_normalization_144/ReadVariableOpì
6sequential_24/batch_normalization_144/ReadVariableOp_1ReadVariableOp?sequential_24_batch_normalization_144_readvariableop_1_resource*
_output_shapes
: *
dtype028
6sequential_24/batch_normalization_144/ReadVariableOp_1
Esequential_24/batch_normalization_144/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_24_batch_normalization_144_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02G
Esequential_24/batch_normalization_144/FusedBatchNormV3/ReadVariableOp
Gsequential_24/batch_normalization_144/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_24_batch_normalization_144_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02I
Gsequential_24/batch_normalization_144/FusedBatchNormV3/ReadVariableOp_1Ú
6sequential_24/batch_normalization_144/FusedBatchNormV3FusedBatchNormV30sequential_24/max_pooling2d_129/MaxPool:output:0<sequential_24/batch_normalization_144/ReadVariableOp:value:0>sequential_24/batch_normalization_144/ReadVariableOp_1:value:0Msequential_24/batch_normalization_144/FusedBatchNormV3/ReadVariableOp:value:0Osequential_24/batch_normalization_144/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 28
6sequential_24/batch_normalization_144/FusedBatchNormV3ã
/sequential_24/conv_layer2/Conv2D/ReadVariableOpReadVariableOp8sequential_24_conv_layer2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype021
/sequential_24/conv_layer2/Conv2D/ReadVariableOp¨
 sequential_24/conv_layer2/Conv2DConv2D:sequential_24/batch_normalization_144/FusedBatchNormV3:y:07sequential_24/conv_layer2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2"
 sequential_24/conv_layer2/Conv2DÚ
0sequential_24/conv_layer2/BiasAdd/ReadVariableOpReadVariableOp9sequential_24_conv_layer2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0sequential_24/conv_layer2/BiasAdd/ReadVariableOpò
!sequential_24/conv_layer2/BiasAddBiasAdd)sequential_24/conv_layer2/Conv2D:output:08sequential_24/conv_layer2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2#
!sequential_24/conv_layer2/BiasAdd°
sequential_24/conv_layer2/ReluRelu*sequential_24/conv_layer2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
sequential_24/conv_layer2/Reluø
'sequential_24/max_pooling2d_130/MaxPoolMaxPool,sequential_24/conv_layer2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@*
ksize
*
paddingVALID*
strides
2)
'sequential_24/max_pooling2d_130/MaxPoolæ
4sequential_24/batch_normalization_145/ReadVariableOpReadVariableOp=sequential_24_batch_normalization_145_readvariableop_resource*
_output_shapes
:@*
dtype026
4sequential_24/batch_normalization_145/ReadVariableOpì
6sequential_24/batch_normalization_145/ReadVariableOp_1ReadVariableOp?sequential_24_batch_normalization_145_readvariableop_1_resource*
_output_shapes
:@*
dtype028
6sequential_24/batch_normalization_145/ReadVariableOp_1
Esequential_24/batch_normalization_145/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_24_batch_normalization_145_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02G
Esequential_24/batch_normalization_145/FusedBatchNormV3/ReadVariableOp
Gsequential_24/batch_normalization_145/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_24_batch_normalization_145_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02I
Gsequential_24/batch_normalization_145/FusedBatchNormV3/ReadVariableOp_1Ø
6sequential_24/batch_normalization_145/FusedBatchNormV3FusedBatchNormV30sequential_24/max_pooling2d_130/MaxPool:output:0<sequential_24/batch_normalization_145/ReadVariableOp:value:0>sequential_24/batch_normalization_145/ReadVariableOp_1:value:0Msequential_24/batch_normalization_145/FusedBatchNormV3/ReadVariableOp:value:0Osequential_24/batch_normalization_145/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿII@:@:@:@:@:*
epsilon%o:*
is_training( 28
6sequential_24/batch_normalization_145/FusedBatchNormV3ä
/sequential_24/conv_layer3/Conv2D/ReadVariableOpReadVariableOp8sequential_24_conv_layer3_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype021
/sequential_24/conv_layer3/Conv2D/ReadVariableOp§
 sequential_24/conv_layer3/Conv2DConv2D:sequential_24/batch_normalization_145/FusedBatchNormV3:y:07sequential_24/conv_layer3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿGG*
paddingVALID*
strides
2"
 sequential_24/conv_layer3/Conv2DÛ
0sequential_24/conv_layer3/BiasAdd/ReadVariableOpReadVariableOp9sequential_24_conv_layer3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0sequential_24/conv_layer3/BiasAdd/ReadVariableOpñ
!sequential_24/conv_layer3/BiasAddBiasAdd)sequential_24/conv_layer3/Conv2D:output:08sequential_24/conv_layer3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿGG2#
!sequential_24/conv_layer3/BiasAdd¯
sequential_24/conv_layer3/ReluRelu*sequential_24/conv_layer3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿGG2 
sequential_24/conv_layer3/Reluù
'sequential_24/max_pooling2d_131/MaxPoolMaxPool,sequential_24/conv_layer3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
ksize
*
paddingVALID*
strides
2)
'sequential_24/max_pooling2d_131/MaxPoolç
4sequential_24/batch_normalization_146/ReadVariableOpReadVariableOp=sequential_24_batch_normalization_146_readvariableop_resource*
_output_shapes	
:*
dtype026
4sequential_24/batch_normalization_146/ReadVariableOpí
6sequential_24/batch_normalization_146/ReadVariableOp_1ReadVariableOp?sequential_24_batch_normalization_146_readvariableop_1_resource*
_output_shapes	
:*
dtype028
6sequential_24/batch_normalization_146/ReadVariableOp_1
Esequential_24/batch_normalization_146/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_24_batch_normalization_146_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02G
Esequential_24/batch_normalization_146/FusedBatchNormV3/ReadVariableOp 
Gsequential_24/batch_normalization_146/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_24_batch_normalization_146_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02I
Gsequential_24/batch_normalization_146/FusedBatchNormV3/ReadVariableOp_1Ý
6sequential_24/batch_normalization_146/FusedBatchNormV3FusedBatchNormV30sequential_24/max_pooling2d_131/MaxPool:output:0<sequential_24/batch_normalization_146/ReadVariableOp:value:0>sequential_24/batch_normalization_146/ReadVariableOp_1:value:0Msequential_24/batch_normalization_146/FusedBatchNormV3/ReadVariableOp:value:0Osequential_24/batch_normalization_146/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ##:::::*
epsilon%o:*
is_training( 28
6sequential_24/batch_normalization_146/FusedBatchNormV3å
/sequential_24/conv_layer4/Conv2D/ReadVariableOpReadVariableOp8sequential_24_conv_layer4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype021
/sequential_24/conv_layer4/Conv2D/ReadVariableOp§
 sequential_24/conv_layer4/Conv2DConv2D:sequential_24/batch_normalization_146/FusedBatchNormV3:y:07sequential_24/conv_layer4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!!*
paddingVALID*
strides
2"
 sequential_24/conv_layer4/Conv2DÛ
0sequential_24/conv_layer4/BiasAdd/ReadVariableOpReadVariableOp9sequential_24_conv_layer4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0sequential_24/conv_layer4/BiasAdd/ReadVariableOpñ
!sequential_24/conv_layer4/BiasAddBiasAdd)sequential_24/conv_layer4/Conv2D:output:08sequential_24/conv_layer4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!!2#
!sequential_24/conv_layer4/BiasAdd¯
sequential_24/conv_layer4/ReluRelu*sequential_24/conv_layer4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!!2 
sequential_24/conv_layer4/Reluù
'sequential_24/max_pooling2d_132/MaxPoolMaxPool,sequential_24/conv_layer4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2)
'sequential_24/max_pooling2d_132/MaxPoolç
4sequential_24/batch_normalization_147/ReadVariableOpReadVariableOp=sequential_24_batch_normalization_147_readvariableop_resource*
_output_shapes	
:*
dtype026
4sequential_24/batch_normalization_147/ReadVariableOpí
6sequential_24/batch_normalization_147/ReadVariableOp_1ReadVariableOp?sequential_24_batch_normalization_147_readvariableop_1_resource*
_output_shapes	
:*
dtype028
6sequential_24/batch_normalization_147/ReadVariableOp_1
Esequential_24/batch_normalization_147/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_24_batch_normalization_147_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02G
Esequential_24/batch_normalization_147/FusedBatchNormV3/ReadVariableOp 
Gsequential_24/batch_normalization_147/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_24_batch_normalization_147_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02I
Gsequential_24/batch_normalization_147/FusedBatchNormV3/ReadVariableOp_1Ý
6sequential_24/batch_normalization_147/FusedBatchNormV3FusedBatchNormV30sequential_24/max_pooling2d_132/MaxPool:output:0<sequential_24/batch_normalization_147/ReadVariableOp:value:0>sequential_24/batch_normalization_147/ReadVariableOp_1:value:0Msequential_24/batch_normalization_147/FusedBatchNormV3/ReadVariableOp:value:0Osequential_24/batch_normalization_147/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 28
6sequential_24/batch_normalization_147/FusedBatchNormV3Õ
@sequential_24/global_average_pooling2d_26/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2B
@sequential_24/global_average_pooling2d_26/Mean/reduction_indices¢
.sequential_24/global_average_pooling2d_26/MeanMean:sequential_24/batch_normalization_147/FusedBatchNormV3:y:0Isequential_24/global_average_pooling2d_26/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_24/global_average_pooling2d_26/Meanß
0sequential_24/Dense_layer1/MatMul/ReadVariableOpReadVariableOp9sequential_24_dense_layer1_matmul_readvariableop_resource*
_output_shapes
:	2*
dtype022
0sequential_24/Dense_layer1/MatMul/ReadVariableOpõ
!sequential_24/Dense_layer1/MatMulMatMul7sequential_24/global_average_pooling2d_26/Mean:output:08sequential_24/Dense_layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22#
!sequential_24/Dense_layer1/MatMulÝ
1sequential_24/Dense_layer1/BiasAdd/ReadVariableOpReadVariableOp:sequential_24_dense_layer1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype023
1sequential_24/Dense_layer1/BiasAdd/ReadVariableOpí
"sequential_24/Dense_layer1/BiasAddBiasAdd+sequential_24/Dense_layer1/MatMul:product:09sequential_24/Dense_layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22$
"sequential_24/Dense_layer1/BiasAdd©
sequential_24/Dense_layer1/ReluRelu+sequential_24/Dense_layer1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22!
sequential_24/Dense_layer1/ReluÞ
0sequential_24/Dense_layer2/MatMul/ReadVariableOpReadVariableOp9sequential_24_dense_layer2_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype022
0sequential_24/Dense_layer2/MatMul/ReadVariableOpë
!sequential_24/Dense_layer2/MatMulMatMul-sequential_24/Dense_layer1/Relu:activations:08sequential_24/Dense_layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2#
!sequential_24/Dense_layer2/MatMulÝ
1sequential_24/Dense_layer2/BiasAdd/ReadVariableOpReadVariableOp:sequential_24_dense_layer2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype023
1sequential_24/Dense_layer2/BiasAdd/ReadVariableOpí
"sequential_24/Dense_layer2/BiasAddBiasAdd+sequential_24/Dense_layer2/MatMul:product:09sequential_24/Dense_layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2$
"sequential_24/Dense_layer2/BiasAdd©
sequential_24/Dense_layer2/ReluRelu+sequential_24/Dense_layer2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2!
sequential_24/Dense_layer2/ReluÞ
0sequential_24/output_layer/MatMul/ReadVariableOpReadVariableOp9sequential_24_output_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype022
0sequential_24/output_layer/MatMul/ReadVariableOpë
!sequential_24/output_layer/MatMulMatMul-sequential_24/Dense_layer2/Relu:activations:08sequential_24/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_24/output_layer/MatMulÝ
1sequential_24/output_layer/BiasAdd/ReadVariableOpReadVariableOp:sequential_24_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_24/output_layer/BiasAdd/ReadVariableOpí
"sequential_24/output_layer/BiasAddBiasAdd+sequential_24/output_layer/MatMul:product:09sequential_24/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"sequential_24/output_layer/BiasAdd²
"sequential_24/output_layer/SoftmaxSoftmax+sequential_24/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"sequential_24/output_layer/Softmax
IdentityIdentity,sequential_24/output_layer/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*º
_input_shapes¨
¥:ÿÿÿÿÿÿÿÿÿ¬¬:::::::::::::::::::::::::::::::::::^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
%
_user_specified_nameinput_layer
	
¯
G__inference_conv_layer2_layer_call_and_return_conditional_losses_210173

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ :::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
è

-__inference_Dense_layer2_layer_call_fn_212219

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Dense_layer2_layer_call_and_return_conditional_losses_2105042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
´
N
2__inference_max_pooling2d_131_layer_call_fn_209751

inputs
identityñ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_131_layer_call_and_return_conditional_losses_2097452
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨
«
8__inference_batch_normalization_147_layer_call_fn_212102

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_147_layer_call_and_return_conditional_losses_2099292
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨
«
8__inference_batch_normalization_146_layer_call_fn_212018

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_146_layer_call_and_return_conditional_losses_2098132
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°·
ï*
__inference__traced_save_212535
file_prefix<
8savev2_batch_normalization_143_gamma_read_readvariableop;
7savev2_batch_normalization_143_beta_read_readvariableopB
>savev2_batch_normalization_143_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_143_moving_variance_read_readvariableop1
-savev2_conv_layer1_kernel_read_readvariableop/
+savev2_conv_layer1_bias_read_readvariableop<
8savev2_batch_normalization_144_gamma_read_readvariableop;
7savev2_batch_normalization_144_beta_read_readvariableopB
>savev2_batch_normalization_144_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_144_moving_variance_read_readvariableop1
-savev2_conv_layer2_kernel_read_readvariableop/
+savev2_conv_layer2_bias_read_readvariableop<
8savev2_batch_normalization_145_gamma_read_readvariableop;
7savev2_batch_normalization_145_beta_read_readvariableopB
>savev2_batch_normalization_145_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_145_moving_variance_read_readvariableop1
-savev2_conv_layer3_kernel_read_readvariableop/
+savev2_conv_layer3_bias_read_readvariableop<
8savev2_batch_normalization_146_gamma_read_readvariableop;
7savev2_batch_normalization_146_beta_read_readvariableopB
>savev2_batch_normalization_146_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_146_moving_variance_read_readvariableop1
-savev2_conv_layer4_kernel_read_readvariableop/
+savev2_conv_layer4_bias_read_readvariableop<
8savev2_batch_normalization_147_gamma_read_readvariableop;
7savev2_batch_normalization_147_beta_read_readvariableopB
>savev2_batch_normalization_147_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_147_moving_variance_read_readvariableop2
.savev2_dense_layer1_kernel_read_readvariableop0
,savev2_dense_layer1_bias_read_readvariableop2
.savev2_dense_layer2_kernel_read_readvariableop0
,savev2_dense_layer2_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopC
?savev2_adam_batch_normalization_143_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_143_beta_m_read_readvariableop8
4savev2_adam_conv_layer1_kernel_m_read_readvariableop6
2savev2_adam_conv_layer1_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_144_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_144_beta_m_read_readvariableop8
4savev2_adam_conv_layer2_kernel_m_read_readvariableop6
2savev2_adam_conv_layer2_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_145_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_145_beta_m_read_readvariableop8
4savev2_adam_conv_layer3_kernel_m_read_readvariableop6
2savev2_adam_conv_layer3_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_146_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_146_beta_m_read_readvariableop8
4savev2_adam_conv_layer4_kernel_m_read_readvariableop6
2savev2_adam_conv_layer4_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_147_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_147_beta_m_read_readvariableop9
5savev2_adam_dense_layer1_kernel_m_read_readvariableop7
3savev2_adam_dense_layer1_bias_m_read_readvariableop9
5savev2_adam_dense_layer2_kernel_m_read_readvariableop7
3savev2_adam_dense_layer2_bias_m_read_readvariableop9
5savev2_adam_output_layer_kernel_m_read_readvariableop7
3savev2_adam_output_layer_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_143_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_143_beta_v_read_readvariableop8
4savev2_adam_conv_layer1_kernel_v_read_readvariableop6
2savev2_adam_conv_layer1_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_144_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_144_beta_v_read_readvariableop8
4savev2_adam_conv_layer2_kernel_v_read_readvariableop6
2savev2_adam_conv_layer2_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_145_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_145_beta_v_read_readvariableop8
4savev2_adam_conv_layer3_kernel_v_read_readvariableop6
2savev2_adam_conv_layer3_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_146_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_146_beta_v_read_readvariableop8
4savev2_adam_conv_layer4_kernel_v_read_readvariableop6
2savev2_adam_conv_layer4_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_147_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_147_beta_v_read_readvariableop9
5savev2_adam_dense_layer1_kernel_v_read_readvariableop7
3savev2_adam_dense_layer1_bias_v_read_readvariableop9
5savev2_adam_dense_layer2_kernel_v_read_readvariableop7
3savev2_adam_dense_layer2_bias_v_read_readvariableop9
5savev2_adam_output_layer_kernel_v_read_readvariableop7
3savev2_adam_output_layer_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_efa73bd3e1244b1fa5f58ed1e9b4e9bb/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename3
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:\*
dtype0*2
value2B2\B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÃ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:\*
dtype0*Í
valueÃBÀ\B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices¬)
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:08savev2_batch_normalization_143_gamma_read_readvariableop7savev2_batch_normalization_143_beta_read_readvariableop>savev2_batch_normalization_143_moving_mean_read_readvariableopBsavev2_batch_normalization_143_moving_variance_read_readvariableop-savev2_conv_layer1_kernel_read_readvariableop+savev2_conv_layer1_bias_read_readvariableop8savev2_batch_normalization_144_gamma_read_readvariableop7savev2_batch_normalization_144_beta_read_readvariableop>savev2_batch_normalization_144_moving_mean_read_readvariableopBsavev2_batch_normalization_144_moving_variance_read_readvariableop-savev2_conv_layer2_kernel_read_readvariableop+savev2_conv_layer2_bias_read_readvariableop8savev2_batch_normalization_145_gamma_read_readvariableop7savev2_batch_normalization_145_beta_read_readvariableop>savev2_batch_normalization_145_moving_mean_read_readvariableopBsavev2_batch_normalization_145_moving_variance_read_readvariableop-savev2_conv_layer3_kernel_read_readvariableop+savev2_conv_layer3_bias_read_readvariableop8savev2_batch_normalization_146_gamma_read_readvariableop7savev2_batch_normalization_146_beta_read_readvariableop>savev2_batch_normalization_146_moving_mean_read_readvariableopBsavev2_batch_normalization_146_moving_variance_read_readvariableop-savev2_conv_layer4_kernel_read_readvariableop+savev2_conv_layer4_bias_read_readvariableop8savev2_batch_normalization_147_gamma_read_readvariableop7savev2_batch_normalization_147_beta_read_readvariableop>savev2_batch_normalization_147_moving_mean_read_readvariableopBsavev2_batch_normalization_147_moving_variance_read_readvariableop.savev2_dense_layer1_kernel_read_readvariableop,savev2_dense_layer1_bias_read_readvariableop.savev2_dense_layer2_kernel_read_readvariableop,savev2_dense_layer2_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop?savev2_adam_batch_normalization_143_gamma_m_read_readvariableop>savev2_adam_batch_normalization_143_beta_m_read_readvariableop4savev2_adam_conv_layer1_kernel_m_read_readvariableop2savev2_adam_conv_layer1_bias_m_read_readvariableop?savev2_adam_batch_normalization_144_gamma_m_read_readvariableop>savev2_adam_batch_normalization_144_beta_m_read_readvariableop4savev2_adam_conv_layer2_kernel_m_read_readvariableop2savev2_adam_conv_layer2_bias_m_read_readvariableop?savev2_adam_batch_normalization_145_gamma_m_read_readvariableop>savev2_adam_batch_normalization_145_beta_m_read_readvariableop4savev2_adam_conv_layer3_kernel_m_read_readvariableop2savev2_adam_conv_layer3_bias_m_read_readvariableop?savev2_adam_batch_normalization_146_gamma_m_read_readvariableop>savev2_adam_batch_normalization_146_beta_m_read_readvariableop4savev2_adam_conv_layer4_kernel_m_read_readvariableop2savev2_adam_conv_layer4_bias_m_read_readvariableop?savev2_adam_batch_normalization_147_gamma_m_read_readvariableop>savev2_adam_batch_normalization_147_beta_m_read_readvariableop5savev2_adam_dense_layer1_kernel_m_read_readvariableop3savev2_adam_dense_layer1_bias_m_read_readvariableop5savev2_adam_dense_layer2_kernel_m_read_readvariableop3savev2_adam_dense_layer2_bias_m_read_readvariableop5savev2_adam_output_layer_kernel_m_read_readvariableop3savev2_adam_output_layer_bias_m_read_readvariableop?savev2_adam_batch_normalization_143_gamma_v_read_readvariableop>savev2_adam_batch_normalization_143_beta_v_read_readvariableop4savev2_adam_conv_layer1_kernel_v_read_readvariableop2savev2_adam_conv_layer1_bias_v_read_readvariableop?savev2_adam_batch_normalization_144_gamma_v_read_readvariableop>savev2_adam_batch_normalization_144_beta_v_read_readvariableop4savev2_adam_conv_layer2_kernel_v_read_readvariableop2savev2_adam_conv_layer2_bias_v_read_readvariableop?savev2_adam_batch_normalization_145_gamma_v_read_readvariableop>savev2_adam_batch_normalization_145_beta_v_read_readvariableop4savev2_adam_conv_layer3_kernel_v_read_readvariableop2savev2_adam_conv_layer3_bias_v_read_readvariableop?savev2_adam_batch_normalization_146_gamma_v_read_readvariableop>savev2_adam_batch_normalization_146_beta_v_read_readvariableop4savev2_adam_conv_layer4_kernel_v_read_readvariableop2savev2_adam_conv_layer4_bias_v_read_readvariableop?savev2_adam_batch_normalization_147_gamma_v_read_readvariableop>savev2_adam_batch_normalization_147_beta_v_read_readvariableop5savev2_adam_dense_layer1_kernel_v_read_readvariableop3savev2_adam_dense_layer1_bias_v_read_readvariableop5savev2_adam_dense_layer2_kernel_v_read_readvariableop3savev2_adam_dense_layer2_bias_v_read_readvariableop5savev2_adam_output_layer_kernel_v_read_readvariableop3savev2_adam_output_layer_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *j
dtypes`
^2\	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*í
_input_shapesÛ
Ø: ::::: : : : : : : @:@:@:@:@:@:@::::::::::::	2:2:2d:d:d:: : : : : : : : : ::: : : : : @:@:@:@:@::::::::	2:2:2d:d:d:::: : : : : @:@:@:@:@::::::::	2:2:2d:d:d:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::%!

_output_shapes
:	2: 

_output_shapes
:2:$ 

_output_shapes

:2d:  

_output_shapes
:d:$! 

_output_shapes

:d: "

_output_shapes
::#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: : ,

_output_shapes
:: -

_output_shapes
::,.(
&
_output_shapes
: : /

_output_shapes
: : 0

_output_shapes
: : 1

_output_shapes
: :,2(
&
_output_shapes
: @: 3

_output_shapes
:@: 4

_output_shapes
:@: 5

_output_shapes
:@:-6)
'
_output_shapes
:@:!7

_output_shapes	
::!8

_output_shapes	
::!9

_output_shapes	
::.:*
(
_output_shapes
::!;

_output_shapes	
::!<

_output_shapes	
::!=

_output_shapes	
::%>!

_output_shapes
:	2: ?

_output_shapes
:2:$@ 

_output_shapes

:2d: A

_output_shapes
:d:$B 

_output_shapes

:d: C

_output_shapes
:: D

_output_shapes
:: E

_output_shapes
::,F(
&
_output_shapes
: : G

_output_shapes
: : H

_output_shapes
: : I

_output_shapes
: :,J(
&
_output_shapes
: @: K

_output_shapes
:@: L

_output_shapes
:@: M

_output_shapes
:@:-N)
'
_output_shapes
:@:!O

_output_shapes	
::!P

_output_shapes	
::!Q

_output_shapes	
::.R*
(
_output_shapes
::!S

_output_shapes	
::!T

_output_shapes	
::!U

_output_shapes	
::%V!

_output_shapes
:	2: W

_output_shapes
:2:$X 

_output_shapes

:2d: Y

_output_shapes
:d:$Z 

_output_shapes

:d: [

_output_shapes
::\

_output_shapes
: 

°
S__inference_batch_normalization_143_layer_call_and_return_conditional_losses_210007

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ú
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ¬¬:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ¬¬::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
 
_user_specified_nameinputs
Â
5
"__inference__traced_restore_212818
file_prefix2
.assignvariableop_batch_normalization_143_gamma3
/assignvariableop_1_batch_normalization_143_beta:
6assignvariableop_2_batch_normalization_143_moving_mean>
:assignvariableop_3_batch_normalization_143_moving_variance)
%assignvariableop_4_conv_layer1_kernel'
#assignvariableop_5_conv_layer1_bias4
0assignvariableop_6_batch_normalization_144_gamma3
/assignvariableop_7_batch_normalization_144_beta:
6assignvariableop_8_batch_normalization_144_moving_mean>
:assignvariableop_9_batch_normalization_144_moving_variance*
&assignvariableop_10_conv_layer2_kernel(
$assignvariableop_11_conv_layer2_bias5
1assignvariableop_12_batch_normalization_145_gamma4
0assignvariableop_13_batch_normalization_145_beta;
7assignvariableop_14_batch_normalization_145_moving_mean?
;assignvariableop_15_batch_normalization_145_moving_variance*
&assignvariableop_16_conv_layer3_kernel(
$assignvariableop_17_conv_layer3_bias5
1assignvariableop_18_batch_normalization_146_gamma4
0assignvariableop_19_batch_normalization_146_beta;
7assignvariableop_20_batch_normalization_146_moving_mean?
;assignvariableop_21_batch_normalization_146_moving_variance*
&assignvariableop_22_conv_layer4_kernel(
$assignvariableop_23_conv_layer4_bias5
1assignvariableop_24_batch_normalization_147_gamma4
0assignvariableop_25_batch_normalization_147_beta;
7assignvariableop_26_batch_normalization_147_moving_mean?
;assignvariableop_27_batch_normalization_147_moving_variance+
'assignvariableop_28_dense_layer1_kernel)
%assignvariableop_29_dense_layer1_bias+
'assignvariableop_30_dense_layer2_kernel)
%assignvariableop_31_dense_layer2_bias+
'assignvariableop_32_output_layer_kernel)
%assignvariableop_33_output_layer_bias!
assignvariableop_34_adam_iter#
assignvariableop_35_adam_beta_1#
assignvariableop_36_adam_beta_2"
assignvariableop_37_adam_decay*
&assignvariableop_38_adam_learning_rate
assignvariableop_39_total
assignvariableop_40_count
assignvariableop_41_total_1
assignvariableop_42_count_1<
8assignvariableop_43_adam_batch_normalization_143_gamma_m;
7assignvariableop_44_adam_batch_normalization_143_beta_m1
-assignvariableop_45_adam_conv_layer1_kernel_m/
+assignvariableop_46_adam_conv_layer1_bias_m<
8assignvariableop_47_adam_batch_normalization_144_gamma_m;
7assignvariableop_48_adam_batch_normalization_144_beta_m1
-assignvariableop_49_adam_conv_layer2_kernel_m/
+assignvariableop_50_adam_conv_layer2_bias_m<
8assignvariableop_51_adam_batch_normalization_145_gamma_m;
7assignvariableop_52_adam_batch_normalization_145_beta_m1
-assignvariableop_53_adam_conv_layer3_kernel_m/
+assignvariableop_54_adam_conv_layer3_bias_m<
8assignvariableop_55_adam_batch_normalization_146_gamma_m;
7assignvariableop_56_adam_batch_normalization_146_beta_m1
-assignvariableop_57_adam_conv_layer4_kernel_m/
+assignvariableop_58_adam_conv_layer4_bias_m<
8assignvariableop_59_adam_batch_normalization_147_gamma_m;
7assignvariableop_60_adam_batch_normalization_147_beta_m2
.assignvariableop_61_adam_dense_layer1_kernel_m0
,assignvariableop_62_adam_dense_layer1_bias_m2
.assignvariableop_63_adam_dense_layer2_kernel_m0
,assignvariableop_64_adam_dense_layer2_bias_m2
.assignvariableop_65_adam_output_layer_kernel_m0
,assignvariableop_66_adam_output_layer_bias_m<
8assignvariableop_67_adam_batch_normalization_143_gamma_v;
7assignvariableop_68_adam_batch_normalization_143_beta_v1
-assignvariableop_69_adam_conv_layer1_kernel_v/
+assignvariableop_70_adam_conv_layer1_bias_v<
8assignvariableop_71_adam_batch_normalization_144_gamma_v;
7assignvariableop_72_adam_batch_normalization_144_beta_v1
-assignvariableop_73_adam_conv_layer2_kernel_v/
+assignvariableop_74_adam_conv_layer2_bias_v<
8assignvariableop_75_adam_batch_normalization_145_gamma_v;
7assignvariableop_76_adam_batch_normalization_145_beta_v1
-assignvariableop_77_adam_conv_layer3_kernel_v/
+assignvariableop_78_adam_conv_layer3_bias_v<
8assignvariableop_79_adam_batch_normalization_146_gamma_v;
7assignvariableop_80_adam_batch_normalization_146_beta_v1
-assignvariableop_81_adam_conv_layer4_kernel_v/
+assignvariableop_82_adam_conv_layer4_bias_v<
8assignvariableop_83_adam_batch_normalization_147_gamma_v;
7assignvariableop_84_adam_batch_normalization_147_beta_v2
.assignvariableop_85_adam_dense_layer1_kernel_v0
,assignvariableop_86_adam_dense_layer1_bias_v2
.assignvariableop_87_adam_dense_layer2_kernel_v0
,assignvariableop_88_adam_dense_layer2_bias_v2
.assignvariableop_89_adam_output_layer_kernel_v0
,assignvariableop_90_adam_output_layer_bias_v
identity_92¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_87¢AssignVariableOp_88¢AssignVariableOp_89¢AssignVariableOp_9¢AssignVariableOp_903
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:\*
dtype0*2
value2B2\B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÉ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:\*
dtype0*Í
valueÃBÀ\B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesú
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesó
ð::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*j
dtypes`
^2\	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity­
AssignVariableOpAssignVariableOp.assignvariableop_batch_normalization_143_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1´
AssignVariableOp_1AssignVariableOp/assignvariableop_1_batch_normalization_143_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2»
AssignVariableOp_2AssignVariableOp6assignvariableop_2_batch_normalization_143_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¿
AssignVariableOp_3AssignVariableOp:assignvariableop_3_batch_normalization_143_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ª
AssignVariableOp_4AssignVariableOp%assignvariableop_4_conv_layer1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¨
AssignVariableOp_5AssignVariableOp#assignvariableop_5_conv_layer1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6µ
AssignVariableOp_6AssignVariableOp0assignvariableop_6_batch_normalization_144_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7´
AssignVariableOp_7AssignVariableOp/assignvariableop_7_batch_normalization_144_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8»
AssignVariableOp_8AssignVariableOp6assignvariableop_8_batch_normalization_144_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¿
AssignVariableOp_9AssignVariableOp:assignvariableop_9_batch_normalization_144_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10®
AssignVariableOp_10AssignVariableOp&assignvariableop_10_conv_layer2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¬
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv_layer2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¹
AssignVariableOp_12AssignVariableOp1assignvariableop_12_batch_normalization_145_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¸
AssignVariableOp_13AssignVariableOp0assignvariableop_13_batch_normalization_145_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¿
AssignVariableOp_14AssignVariableOp7assignvariableop_14_batch_normalization_145_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ã
AssignVariableOp_15AssignVariableOp;assignvariableop_15_batch_normalization_145_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16®
AssignVariableOp_16AssignVariableOp&assignvariableop_16_conv_layer3_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¬
AssignVariableOp_17AssignVariableOp$assignvariableop_17_conv_layer3_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¹
AssignVariableOp_18AssignVariableOp1assignvariableop_18_batch_normalization_146_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¸
AssignVariableOp_19AssignVariableOp0assignvariableop_19_batch_normalization_146_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¿
AssignVariableOp_20AssignVariableOp7assignvariableop_20_batch_normalization_146_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ã
AssignVariableOp_21AssignVariableOp;assignvariableop_21_batch_normalization_146_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22®
AssignVariableOp_22AssignVariableOp&assignvariableop_22_conv_layer4_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¬
AssignVariableOp_23AssignVariableOp$assignvariableop_23_conv_layer4_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¹
AssignVariableOp_24AssignVariableOp1assignvariableop_24_batch_normalization_147_gammaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¸
AssignVariableOp_25AssignVariableOp0assignvariableop_25_batch_normalization_147_betaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¿
AssignVariableOp_26AssignVariableOp7assignvariableop_26_batch_normalization_147_moving_meanIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ã
AssignVariableOp_27AssignVariableOp;assignvariableop_27_batch_normalization_147_moving_varianceIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28¯
AssignVariableOp_28AssignVariableOp'assignvariableop_28_dense_layer1_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29­
AssignVariableOp_29AssignVariableOp%assignvariableop_29_dense_layer1_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¯
AssignVariableOp_30AssignVariableOp'assignvariableop_30_dense_layer2_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31­
AssignVariableOp_31AssignVariableOp%assignvariableop_31_dense_layer2_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32¯
AssignVariableOp_32AssignVariableOp'assignvariableop_32_output_layer_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33­
AssignVariableOp_33AssignVariableOp%assignvariableop_33_output_layer_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_34¥
AssignVariableOp_34AssignVariableOpassignvariableop_34_adam_iterIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35§
AssignVariableOp_35AssignVariableOpassignvariableop_35_adam_beta_1Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36§
AssignVariableOp_36AssignVariableOpassignvariableop_36_adam_beta_2Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37¦
AssignVariableOp_37AssignVariableOpassignvariableop_37_adam_decayIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38®
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_learning_rateIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39¡
AssignVariableOp_39AssignVariableOpassignvariableop_39_totalIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40¡
AssignVariableOp_40AssignVariableOpassignvariableop_40_countIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41£
AssignVariableOp_41AssignVariableOpassignvariableop_41_total_1Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42£
AssignVariableOp_42AssignVariableOpassignvariableop_42_count_1Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43À
AssignVariableOp_43AssignVariableOp8assignvariableop_43_adam_batch_normalization_143_gamma_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44¿
AssignVariableOp_44AssignVariableOp7assignvariableop_44_adam_batch_normalization_143_beta_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45µ
AssignVariableOp_45AssignVariableOp-assignvariableop_45_adam_conv_layer1_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46³
AssignVariableOp_46AssignVariableOp+assignvariableop_46_adam_conv_layer1_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47À
AssignVariableOp_47AssignVariableOp8assignvariableop_47_adam_batch_normalization_144_gamma_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48¿
AssignVariableOp_48AssignVariableOp7assignvariableop_48_adam_batch_normalization_144_beta_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49µ
AssignVariableOp_49AssignVariableOp-assignvariableop_49_adam_conv_layer2_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50³
AssignVariableOp_50AssignVariableOp+assignvariableop_50_adam_conv_layer2_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51À
AssignVariableOp_51AssignVariableOp8assignvariableop_51_adam_batch_normalization_145_gamma_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52¿
AssignVariableOp_52AssignVariableOp7assignvariableop_52_adam_batch_normalization_145_beta_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53µ
AssignVariableOp_53AssignVariableOp-assignvariableop_53_adam_conv_layer3_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54³
AssignVariableOp_54AssignVariableOp+assignvariableop_54_adam_conv_layer3_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55À
AssignVariableOp_55AssignVariableOp8assignvariableop_55_adam_batch_normalization_146_gamma_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56¿
AssignVariableOp_56AssignVariableOp7assignvariableop_56_adam_batch_normalization_146_beta_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57µ
AssignVariableOp_57AssignVariableOp-assignvariableop_57_adam_conv_layer4_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58³
AssignVariableOp_58AssignVariableOp+assignvariableop_58_adam_conv_layer4_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59À
AssignVariableOp_59AssignVariableOp8assignvariableop_59_adam_batch_normalization_147_gamma_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60¿
AssignVariableOp_60AssignVariableOp7assignvariableop_60_adam_batch_normalization_147_beta_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61¶
AssignVariableOp_61AssignVariableOp.assignvariableop_61_adam_dense_layer1_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62´
AssignVariableOp_62AssignVariableOp,assignvariableop_62_adam_dense_layer1_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63¶
AssignVariableOp_63AssignVariableOp.assignvariableop_63_adam_dense_layer2_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64´
AssignVariableOp_64AssignVariableOp,assignvariableop_64_adam_dense_layer2_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65¶
AssignVariableOp_65AssignVariableOp.assignvariableop_65_adam_output_layer_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66´
AssignVariableOp_66AssignVariableOp,assignvariableop_66_adam_output_layer_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67À
AssignVariableOp_67AssignVariableOp8assignvariableop_67_adam_batch_normalization_143_gamma_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68¿
AssignVariableOp_68AssignVariableOp7assignvariableop_68_adam_batch_normalization_143_beta_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69µ
AssignVariableOp_69AssignVariableOp-assignvariableop_69_adam_conv_layer1_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70³
AssignVariableOp_70AssignVariableOp+assignvariableop_70_adam_conv_layer1_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71À
AssignVariableOp_71AssignVariableOp8assignvariableop_71_adam_batch_normalization_144_gamma_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72¿
AssignVariableOp_72AssignVariableOp7assignvariableop_72_adam_batch_normalization_144_beta_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73µ
AssignVariableOp_73AssignVariableOp-assignvariableop_73_adam_conv_layer2_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74³
AssignVariableOp_74AssignVariableOp+assignvariableop_74_adam_conv_layer2_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75À
AssignVariableOp_75AssignVariableOp8assignvariableop_75_adam_batch_normalization_145_gamma_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76¿
AssignVariableOp_76AssignVariableOp7assignvariableop_76_adam_batch_normalization_145_beta_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77µ
AssignVariableOp_77AssignVariableOp-assignvariableop_77_adam_conv_layer3_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78³
AssignVariableOp_78AssignVariableOp+assignvariableop_78_adam_conv_layer3_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79À
AssignVariableOp_79AssignVariableOp8assignvariableop_79_adam_batch_normalization_146_gamma_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80¿
AssignVariableOp_80AssignVariableOp7assignvariableop_80_adam_batch_normalization_146_beta_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81µ
AssignVariableOp_81AssignVariableOp-assignvariableop_81_adam_conv_layer4_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82³
AssignVariableOp_82AssignVariableOp+assignvariableop_82_adam_conv_layer4_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83À
AssignVariableOp_83AssignVariableOp8assignvariableop_83_adam_batch_normalization_147_gamma_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84¿
AssignVariableOp_84AssignVariableOp7assignvariableop_84_adam_batch_normalization_147_beta_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85¶
AssignVariableOp_85AssignVariableOp.assignvariableop_85_adam_dense_layer1_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86´
AssignVariableOp_86AssignVariableOp,assignvariableop_86_adam_dense_layer1_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87¶
AssignVariableOp_87AssignVariableOp.assignvariableop_87_adam_dense_layer2_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88´
AssignVariableOp_88AssignVariableOp,assignvariableop_88_adam_dense_layer2_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89¶
AssignVariableOp_89AssignVariableOp.assignvariableop_89_adam_output_layer_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90´
AssignVariableOp_90AssignVariableOp,assignvariableop_90_adam_output_layer_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_909
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp°
Identity_91Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_91£
Identity_92IdentityIdentity_91:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90*
T0*
_output_shapes
: 2
Identity_92"#
identity_92Identity_92:output:0*
_input_shapesñ
î: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_90:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix


S__inference_batch_normalization_145_layer_call_and_return_conditional_losses_209728

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
à
«
8__inference_batch_normalization_147_layer_call_fn_212166

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_147_layer_call_and_return_conditional_losses_2104112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à
«
8__inference_batch_normalization_146_layer_call_fn_211954

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_146_layer_call_and_return_conditional_losses_2103102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ##::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##
 
_user_specified_nameinputs
¦
«
8__inference_batch_normalization_143_layer_call_fn_211523

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallº
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_143_layer_call_and_return_conditional_losses_2094962
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
°
S__inference_batch_normalization_144_layer_call_and_return_conditional_losses_209581

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¤
«
8__inference_batch_normalization_144_layer_call_fn_211658

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_144_layer_call_and_return_conditional_losses_2095812
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
´
N
2__inference_max_pooling2d_129_layer_call_fn_209519

inputs
identityñ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_129_layer_call_and_return_conditional_losses_2095132
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú

S__inference_batch_normalization_144_layer_call_and_return_conditional_losses_210126

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3r
IdentityIdentityFusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ :::::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Þ

S__inference_batch_normalization_147_layer_call_and_return_conditional_losses_212153

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
°
S__inference_batch_normalization_147_layer_call_and_return_conditional_losses_209929

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
«
8__inference_batch_normalization_143_layer_call_fn_211587

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_143_layer_call_and_return_conditional_losses_2100252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ¬¬::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
 
_user_specified_nameinputs
¥
ú
.__inference_sequential_24_layer_call_fn_211386

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*:
_read_only_resource_inputs
 !"*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_24_layer_call_and_return_conditional_losses_2107292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*º
_input_shapes¨
¥:ÿÿÿÿÿÿÿÿÿ¬¬::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
 
_user_specified_nameinputs
â
«
8__inference_batch_normalization_146_layer_call_fn_211967

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_146_layer_call_and_return_conditional_losses_2103282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ##::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##
 
_user_specified_nameinputs
Û
÷
I__inference_sequential_24_layer_call_and_return_conditional_losses_211313

inputs3
/batch_normalization_143_readvariableop_resource5
1batch_normalization_143_readvariableop_1_resourceD
@batch_normalization_143_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_143_fusedbatchnormv3_readvariableop_1_resource.
*conv_layer1_conv2d_readvariableop_resource/
+conv_layer1_biasadd_readvariableop_resource3
/batch_normalization_144_readvariableop_resource5
1batch_normalization_144_readvariableop_1_resourceD
@batch_normalization_144_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_144_fusedbatchnormv3_readvariableop_1_resource.
*conv_layer2_conv2d_readvariableop_resource/
+conv_layer2_biasadd_readvariableop_resource3
/batch_normalization_145_readvariableop_resource5
1batch_normalization_145_readvariableop_1_resourceD
@batch_normalization_145_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_145_fusedbatchnormv3_readvariableop_1_resource.
*conv_layer3_conv2d_readvariableop_resource/
+conv_layer3_biasadd_readvariableop_resource3
/batch_normalization_146_readvariableop_resource5
1batch_normalization_146_readvariableop_1_resourceD
@batch_normalization_146_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_146_fusedbatchnormv3_readvariableop_1_resource.
*conv_layer4_conv2d_readvariableop_resource/
+conv_layer4_biasadd_readvariableop_resource3
/batch_normalization_147_readvariableop_resource5
1batch_normalization_147_readvariableop_1_resourceD
@batch_normalization_147_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_147_fusedbatchnormv3_readvariableop_1_resource/
+dense_layer1_matmul_readvariableop_resource0
,dense_layer1_biasadd_readvariableop_resource/
+dense_layer2_matmul_readvariableop_resource0
,dense_layer2_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identity¼
&batch_normalization_143/ReadVariableOpReadVariableOp/batch_normalization_143_readvariableop_resource*
_output_shapes
:*
dtype02(
&batch_normalization_143/ReadVariableOpÂ
(batch_normalization_143/ReadVariableOp_1ReadVariableOp1batch_normalization_143_readvariableop_1_resource*
_output_shapes
:*
dtype02*
(batch_normalization_143/ReadVariableOp_1ï
7batch_normalization_143/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_143_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_143/FusedBatchNormV3/ReadVariableOpõ
9batch_normalization_143/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_143_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9batch_normalization_143/FusedBatchNormV3/ReadVariableOp_1Ü
(batch_normalization_143/FusedBatchNormV3FusedBatchNormV3inputs.batch_normalization_143/ReadVariableOp:value:00batch_normalization_143/ReadVariableOp_1:value:0?batch_normalization_143/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_143/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ¬¬:::::*
epsilon%o:*
is_training( 2*
(batch_normalization_143/FusedBatchNormV3¹
!conv_layer1/Conv2D/ReadVariableOpReadVariableOp*conv_layer1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02#
!conv_layer1/Conv2D/ReadVariableOpð
conv_layer1/Conv2DConv2D,batch_normalization_143/FusedBatchNormV3:y:0)conv_layer1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©© *
paddingVALID*
strides
2
conv_layer1/Conv2D°
"conv_layer1/BiasAdd/ReadVariableOpReadVariableOp+conv_layer1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"conv_layer1/BiasAdd/ReadVariableOpº
conv_layer1/BiasAddBiasAddconv_layer1/Conv2D:output:0*conv_layer1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©© 2
conv_layer1/BiasAdd
conv_layer1/ReluReluconv_layer1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©© 2
conv_layer1/ReluÐ
max_pooling2d_129/MaxPoolMaxPoolconv_layer1/Relu:activations:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_129/MaxPool¼
&batch_normalization_144/ReadVariableOpReadVariableOp/batch_normalization_144_readvariableop_resource*
_output_shapes
: *
dtype02(
&batch_normalization_144/ReadVariableOpÂ
(batch_normalization_144/ReadVariableOp_1ReadVariableOp1batch_normalization_144_readvariableop_1_resource*
_output_shapes
: *
dtype02*
(batch_normalization_144/ReadVariableOp_1ï
7batch_normalization_144/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_144_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype029
7batch_normalization_144/FusedBatchNormV3/ReadVariableOpõ
9batch_normalization_144/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_144_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02;
9batch_normalization_144/FusedBatchNormV3/ReadVariableOp_1ø
(batch_normalization_144/FusedBatchNormV3FusedBatchNormV3"max_pooling2d_129/MaxPool:output:0.batch_normalization_144/ReadVariableOp:value:00batch_normalization_144/ReadVariableOp_1:value:0?batch_normalization_144/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_144/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2*
(batch_normalization_144/FusedBatchNormV3¹
!conv_layer2/Conv2D/ReadVariableOpReadVariableOp*conv_layer2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02#
!conv_layer2/Conv2D/ReadVariableOpð
conv_layer2/Conv2DConv2D,batch_normalization_144/FusedBatchNormV3:y:0)conv_layer2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv_layer2/Conv2D°
"conv_layer2/BiasAdd/ReadVariableOpReadVariableOp+conv_layer2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"conv_layer2/BiasAdd/ReadVariableOpº
conv_layer2/BiasAddBiasAddconv_layer2/Conv2D:output:0*conv_layer2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv_layer2/BiasAdd
conv_layer2/ReluReluconv_layer2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv_layer2/ReluÎ
max_pooling2d_130/MaxPoolMaxPoolconv_layer2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_130/MaxPool¼
&batch_normalization_145/ReadVariableOpReadVariableOp/batch_normalization_145_readvariableop_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_145/ReadVariableOpÂ
(batch_normalization_145/ReadVariableOp_1ReadVariableOp1batch_normalization_145_readvariableop_1_resource*
_output_shapes
:@*
dtype02*
(batch_normalization_145/ReadVariableOp_1ï
7batch_normalization_145/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_145_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype029
7batch_normalization_145/FusedBatchNormV3/ReadVariableOpõ
9batch_normalization_145/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_145_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02;
9batch_normalization_145/FusedBatchNormV3/ReadVariableOp_1ö
(batch_normalization_145/FusedBatchNormV3FusedBatchNormV3"max_pooling2d_130/MaxPool:output:0.batch_normalization_145/ReadVariableOp:value:00batch_normalization_145/ReadVariableOp_1:value:0?batch_normalization_145/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_145/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿII@:@:@:@:@:*
epsilon%o:*
is_training( 2*
(batch_normalization_145/FusedBatchNormV3º
!conv_layer3/Conv2D/ReadVariableOpReadVariableOp*conv_layer3_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02#
!conv_layer3/Conv2D/ReadVariableOpï
conv_layer3/Conv2DConv2D,batch_normalization_145/FusedBatchNormV3:y:0)conv_layer3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿGG*
paddingVALID*
strides
2
conv_layer3/Conv2D±
"conv_layer3/BiasAdd/ReadVariableOpReadVariableOp+conv_layer3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"conv_layer3/BiasAdd/ReadVariableOp¹
conv_layer3/BiasAddBiasAddconv_layer3/Conv2D:output:0*conv_layer3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿGG2
conv_layer3/BiasAdd
conv_layer3/ReluReluconv_layer3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿGG2
conv_layer3/ReluÏ
max_pooling2d_131/MaxPoolMaxPoolconv_layer3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
ksize
*
paddingVALID*
strides
2
max_pooling2d_131/MaxPool½
&batch_normalization_146/ReadVariableOpReadVariableOp/batch_normalization_146_readvariableop_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_146/ReadVariableOpÃ
(batch_normalization_146/ReadVariableOp_1ReadVariableOp1batch_normalization_146_readvariableop_1_resource*
_output_shapes	
:*
dtype02*
(batch_normalization_146/ReadVariableOp_1ð
7batch_normalization_146/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_146_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype029
7batch_normalization_146/FusedBatchNormV3/ReadVariableOpö
9batch_normalization_146/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_146_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02;
9batch_normalization_146/FusedBatchNormV3/ReadVariableOp_1û
(batch_normalization_146/FusedBatchNormV3FusedBatchNormV3"max_pooling2d_131/MaxPool:output:0.batch_normalization_146/ReadVariableOp:value:00batch_normalization_146/ReadVariableOp_1:value:0?batch_normalization_146/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_146/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ##:::::*
epsilon%o:*
is_training( 2*
(batch_normalization_146/FusedBatchNormV3»
!conv_layer4/Conv2D/ReadVariableOpReadVariableOp*conv_layer4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02#
!conv_layer4/Conv2D/ReadVariableOpï
conv_layer4/Conv2DConv2D,batch_normalization_146/FusedBatchNormV3:y:0)conv_layer4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!!*
paddingVALID*
strides
2
conv_layer4/Conv2D±
"conv_layer4/BiasAdd/ReadVariableOpReadVariableOp+conv_layer4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"conv_layer4/BiasAdd/ReadVariableOp¹
conv_layer4/BiasAddBiasAddconv_layer4/Conv2D:output:0*conv_layer4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!!2
conv_layer4/BiasAdd
conv_layer4/ReluReluconv_layer4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!!2
conv_layer4/ReluÏ
max_pooling2d_132/MaxPoolMaxPoolconv_layer4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_132/MaxPool½
&batch_normalization_147/ReadVariableOpReadVariableOp/batch_normalization_147_readvariableop_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_147/ReadVariableOpÃ
(batch_normalization_147/ReadVariableOp_1ReadVariableOp1batch_normalization_147_readvariableop_1_resource*
_output_shapes	
:*
dtype02*
(batch_normalization_147/ReadVariableOp_1ð
7batch_normalization_147/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_147_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype029
7batch_normalization_147/FusedBatchNormV3/ReadVariableOpö
9batch_normalization_147/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_147_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02;
9batch_normalization_147/FusedBatchNormV3/ReadVariableOp_1û
(batch_normalization_147/FusedBatchNormV3FusedBatchNormV3"max_pooling2d_132/MaxPool:output:0.batch_normalization_147/ReadVariableOp:value:00batch_normalization_147/ReadVariableOp_1:value:0?batch_normalization_147/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_147/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2*
(batch_normalization_147/FusedBatchNormV3¹
2global_average_pooling2d_26/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      24
2global_average_pooling2d_26/Mean/reduction_indicesê
 global_average_pooling2d_26/MeanMean,batch_normalization_147/FusedBatchNormV3:y:0;global_average_pooling2d_26/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 global_average_pooling2d_26/Meanµ
"Dense_layer1/MatMul/ReadVariableOpReadVariableOp+dense_layer1_matmul_readvariableop_resource*
_output_shapes
:	2*
dtype02$
"Dense_layer1/MatMul/ReadVariableOp½
Dense_layer1/MatMulMatMul)global_average_pooling2d_26/Mean:output:0*Dense_layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
Dense_layer1/MatMul³
#Dense_layer1/BiasAdd/ReadVariableOpReadVariableOp,dense_layer1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02%
#Dense_layer1/BiasAdd/ReadVariableOpµ
Dense_layer1/BiasAddBiasAddDense_layer1/MatMul:product:0+Dense_layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
Dense_layer1/BiasAdd
Dense_layer1/ReluReluDense_layer1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
Dense_layer1/Relu´
"Dense_layer2/MatMul/ReadVariableOpReadVariableOp+dense_layer2_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02$
"Dense_layer2/MatMul/ReadVariableOp³
Dense_layer2/MatMulMatMulDense_layer1/Relu:activations:0*Dense_layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Dense_layer2/MatMul³
#Dense_layer2/BiasAdd/ReadVariableOpReadVariableOp,dense_layer2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02%
#Dense_layer2/BiasAdd/ReadVariableOpµ
Dense_layer2/BiasAddBiasAddDense_layer2/MatMul:product:0+Dense_layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Dense_layer2/BiasAdd
Dense_layer2/ReluReluDense_layer2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Dense_layer2/Relu´
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02$
"output_layer/MatMul/ReadVariableOp³
output_layer/MatMulMatMulDense_layer2/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output_layer/MatMul³
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#output_layer/BiasAdd/ReadVariableOpµ
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output_layer/BiasAdd
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output_layer/Softmaxr
IdentityIdentityoutput_layer/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*º
_input_shapes¨
¥:ÿÿÿÿÿÿÿÿÿ¬¬:::::::::::::::::::::::::::::::::::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
 
_user_specified_nameinputs
Ü
«
8__inference_batch_normalization_145_layer_call_fn_211870

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_145_layer_call_and_return_conditional_losses_2102092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿII@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@
 
_user_specified_nameinputs


S__inference_batch_normalization_143_layer_call_and_return_conditional_losses_211497

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§

S__inference_batch_normalization_147_layer_call_and_return_conditional_losses_212089

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


,__inference_conv_layer4_layer_call_fn_212051

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_layer4_layer_call_and_return_conditional_losses_2103752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!!2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ##::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##
 
_user_specified_nameinputs
µ
°
H__inference_output_layer_layer_call_and_return_conditional_losses_212230

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
â
«
8__inference_batch_normalization_147_layer_call_fn_212179

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_147_layer_call_and_return_conditional_losses_2104292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§

S__inference_batch_normalization_147_layer_call_and_return_conditional_losses_209960

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


S__inference_batch_normalization_144_layer_call_and_return_conditional_losses_211645

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¦
«
8__inference_batch_normalization_145_layer_call_fn_211819

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallº
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_145_layer_call_and_return_conditional_losses_2097282
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¯
ú
.__inference_sequential_24_layer_call_fn_211459

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32
identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_24_layer_call_and_return_conditional_losses_2108912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*º
_input_shapes¨
¥:ÿÿÿÿÿÿÿÿÿ¬¬::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
 
_user_specified_nameinputs
Ò

S__inference_batch_normalization_145_layer_call_and_return_conditional_losses_210227

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿII@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿII@:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@
 
_user_specified_nameinputs
æ^

I__inference_sequential_24_layer_call_and_return_conditional_losses_210548
input_layer"
batch_normalization_143_210052"
batch_normalization_143_210054"
batch_normalization_143_210056"
batch_normalization_143_210058
conv_layer1_210083
conv_layer1_210085"
batch_normalization_144_210153"
batch_normalization_144_210155"
batch_normalization_144_210157"
batch_normalization_144_210159
conv_layer2_210184
conv_layer2_210186"
batch_normalization_145_210254"
batch_normalization_145_210256"
batch_normalization_145_210258"
batch_normalization_145_210260
conv_layer3_210285
conv_layer3_210287"
batch_normalization_146_210355"
batch_normalization_146_210357"
batch_normalization_146_210359"
batch_normalization_146_210361
conv_layer4_210386
conv_layer4_210388"
batch_normalization_147_210456"
batch_normalization_147_210458"
batch_normalization_147_210460"
batch_normalization_147_210462
dense_layer1_210488
dense_layer1_210490
dense_layer2_210515
dense_layer2_210517
output_layer_210542
output_layer_210544
identity¢$Dense_layer1/StatefulPartitionedCall¢$Dense_layer2/StatefulPartitionedCall¢/batch_normalization_143/StatefulPartitionedCall¢/batch_normalization_144/StatefulPartitionedCall¢/batch_normalization_145/StatefulPartitionedCall¢/batch_normalization_146/StatefulPartitionedCall¢/batch_normalization_147/StatefulPartitionedCall¢#conv_layer1/StatefulPartitionedCall¢#conv_layer2/StatefulPartitionedCall¢#conv_layer3/StatefulPartitionedCall¢#conv_layer4/StatefulPartitionedCall¢$output_layer/StatefulPartitionedCall³
/batch_normalization_143/StatefulPartitionedCallStatefulPartitionedCallinput_layerbatch_normalization_143_210052batch_normalization_143_210054batch_normalization_143_210056batch_normalization_143_210058*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_143_layer_call_and_return_conditional_losses_21000721
/batch_normalization_143/StatefulPartitionedCallâ
#conv_layer1/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_143/StatefulPartitionedCall:output:0conv_layer1_210083conv_layer1_210085*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©© *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_layer1_layer_call_and_return_conditional_losses_2100722%
#conv_layer1/StatefulPartitionedCall¢
!max_pooling2d_129/PartitionedCallPartitionedCall,conv_layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_129_layer_call_and_return_conditional_losses_2095132#
!max_pooling2d_129/PartitionedCallÒ
/batch_normalization_144/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_129/PartitionedCall:output:0batch_normalization_144_210153batch_normalization_144_210155batch_normalization_144_210157batch_normalization_144_210159*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_144_layer_call_and_return_conditional_losses_21010821
/batch_normalization_144/StatefulPartitionedCallâ
#conv_layer2/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_144/StatefulPartitionedCall:output:0conv_layer2_210184conv_layer2_210186*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_layer2_layer_call_and_return_conditional_losses_2101732%
#conv_layer2/StatefulPartitionedCall 
!max_pooling2d_130/PartitionedCallPartitionedCall,conv_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_130_layer_call_and_return_conditional_losses_2096292#
!max_pooling2d_130/PartitionedCallÐ
/batch_normalization_145/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_130/PartitionedCall:output:0batch_normalization_145_210254batch_normalization_145_210256batch_normalization_145_210258batch_normalization_145_210260*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_145_layer_call_and_return_conditional_losses_21020921
/batch_normalization_145/StatefulPartitionedCallá
#conv_layer3/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_145/StatefulPartitionedCall:output:0conv_layer3_210285conv_layer3_210287*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿGG*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_layer3_layer_call_and_return_conditional_losses_2102742%
#conv_layer3/StatefulPartitionedCall¡
!max_pooling2d_131/PartitionedCallPartitionedCall,conv_layer3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_131_layer_call_and_return_conditional_losses_2097452#
!max_pooling2d_131/PartitionedCallÑ
/batch_normalization_146/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_131/PartitionedCall:output:0batch_normalization_146_210355batch_normalization_146_210357batch_normalization_146_210359batch_normalization_146_210361*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_146_layer_call_and_return_conditional_losses_21031021
/batch_normalization_146/StatefulPartitionedCallá
#conv_layer4/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_146/StatefulPartitionedCall:output:0conv_layer4_210386conv_layer4_210388*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_layer4_layer_call_and_return_conditional_losses_2103752%
#conv_layer4/StatefulPartitionedCall¡
!max_pooling2d_132/PartitionedCallPartitionedCall,conv_layer4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_132_layer_call_and_return_conditional_losses_2098612#
!max_pooling2d_132/PartitionedCallÑ
/batch_normalization_147/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_132/PartitionedCall:output:0batch_normalization_147_210456batch_normalization_147_210458batch_normalization_147_210460batch_normalization_147_210462*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_147_layer_call_and_return_conditional_losses_21041121
/batch_normalization_147/StatefulPartitionedCallÃ
+global_average_pooling2d_26/PartitionedCallPartitionedCall8batch_normalization_147/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling2d_26_layer_call_and_return_conditional_losses_2099782-
+global_average_pooling2d_26/PartitionedCallÙ
$Dense_layer1/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling2d_26/PartitionedCall:output:0dense_layer1_210488dense_layer1_210490*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Dense_layer1_layer_call_and_return_conditional_losses_2104772&
$Dense_layer1/StatefulPartitionedCallÒ
$Dense_layer2/StatefulPartitionedCallStatefulPartitionedCall-Dense_layer1/StatefulPartitionedCall:output:0dense_layer2_210515dense_layer2_210517*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Dense_layer2_layer_call_and_return_conditional_losses_2105042&
$Dense_layer2/StatefulPartitionedCallÒ
$output_layer/StatefulPartitionedCallStatefulPartitionedCall-Dense_layer2/StatefulPartitionedCall:output:0output_layer_210542output_layer_210544*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_2105312&
$output_layer/StatefulPartitionedCall
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0%^Dense_layer1/StatefulPartitionedCall%^Dense_layer2/StatefulPartitionedCall0^batch_normalization_143/StatefulPartitionedCall0^batch_normalization_144/StatefulPartitionedCall0^batch_normalization_145/StatefulPartitionedCall0^batch_normalization_146/StatefulPartitionedCall0^batch_normalization_147/StatefulPartitionedCall$^conv_layer1/StatefulPartitionedCall$^conv_layer2/StatefulPartitionedCall$^conv_layer3/StatefulPartitionedCall$^conv_layer4/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*º
_input_shapes¨
¥:ÿÿÿÿÿÿÿÿÿ¬¬::::::::::::::::::::::::::::::::::2L
$Dense_layer1/StatefulPartitionedCall$Dense_layer1/StatefulPartitionedCall2L
$Dense_layer2/StatefulPartitionedCall$Dense_layer2/StatefulPartitionedCall2b
/batch_normalization_143/StatefulPartitionedCall/batch_normalization_143/StatefulPartitionedCall2b
/batch_normalization_144/StatefulPartitionedCall/batch_normalization_144/StatefulPartitionedCall2b
/batch_normalization_145/StatefulPartitionedCall/batch_normalization_145/StatefulPartitionedCall2b
/batch_normalization_146/StatefulPartitionedCall/batch_normalization_146/StatefulPartitionedCall2b
/batch_normalization_147/StatefulPartitionedCall/batch_normalization_147/StatefulPartitionedCall2J
#conv_layer1/StatefulPartitionedCall#conv_layer1/StatefulPartitionedCall2J
#conv_layer2/StatefulPartitionedCall#conv_layer2/StatefulPartitionedCall2J
#conv_layer3/StatefulPartitionedCall#conv_layer3/StatefulPartitionedCall2J
#conv_layer4/StatefulPartitionedCall#conv_layer4/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
%
_user_specified_nameinput_layer
	
¯
G__inference_conv_layer4_layer_call_and_return_conditional_losses_212042

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!!*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!!2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!!2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!!2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ##:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##
 
_user_specified_nameinputs
ä
«
8__inference_batch_normalization_144_layer_call_fn_211722

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_144_layer_call_and_return_conditional_losses_2101082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	
¯
G__inference_conv_layer2_layer_call_and_return_conditional_losses_211746

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ :::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
°
°
H__inference_Dense_layer1_layer_call_and_return_conditional_losses_212190

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
°
S__inference_batch_normalization_147_layer_call_and_return_conditional_losses_212071

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
è

-__inference_output_layer_layer_call_fn_212239

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_2105312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
°
°
H__inference_Dense_layer1_layer_call_and_return_conditional_losses_210477

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


S__inference_batch_normalization_144_layer_call_and_return_conditional_losses_209612

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ê

-__inference_Dense_layer1_layer_call_fn_212199

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Dense_layer1_layer_call_and_return_conditional_losses_2104772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§

S__inference_batch_normalization_146_layer_call_and_return_conditional_losses_209844

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_132_layer_call_and_return_conditional_losses_209861

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
N
2__inference_max_pooling2d_132_layer_call_fn_209867

inputs
identityñ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_132_layer_call_and_return_conditional_losses_2098612
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

°
S__inference_batch_normalization_146_layer_call_and_return_conditional_losses_211923

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ##:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ##::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##
 
_user_specified_nameinputs
Þ

S__inference_batch_normalization_146_layer_call_and_return_conditional_losses_210328

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ##:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ##:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##
 
_user_specified_nameinputs

°
S__inference_batch_normalization_146_layer_call_and_return_conditional_losses_210310

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ##:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ##::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##
 
_user_specified_nameinputs
½
s
W__inference_global_average_pooling2d_26_layer_call_and_return_conditional_losses_209978

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
°
H__inference_output_layer_layer_call_and_return_conditional_losses_210531

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
	
¯
G__inference_conv_layer3_layer_call_and_return_conditional_losses_210274

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿGG*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿGG2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿGG2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿGG2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿII@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@
 
_user_specified_nameinputs

°
S__inference_batch_normalization_145_layer_call_and_return_conditional_losses_210209

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿII@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿII@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@
 
_user_specified_nameinputs
Ë
°
S__inference_batch_normalization_143_layer_call_and_return_conditional_losses_209465

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_131_layer_call_and_return_conditional_losses_209745

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤
«
8__inference_batch_normalization_145_layer_call_fn_211806

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_145_layer_call_and_return_conditional_losses_2096972
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ë
°
S__inference_batch_normalization_145_layer_call_and_return_conditional_losses_209697

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ú

S__inference_batch_normalization_144_layer_call_and_return_conditional_losses_211709

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3r
IdentityIdentityFusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ :::::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


,__inference_conv_layer2_layer_call_fn_211755

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_layer2_layer_call_and_return_conditional_losses_2101732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
×
°
S__inference_batch_normalization_146_layer_call_and_return_conditional_losses_211987

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
¯
G__inference_conv_layer1_layer_call_and_return_conditional_losses_211598

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©© *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©© 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©© 2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©© 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ¬¬:::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
 
_user_specified_nameinputs
¤
«
8__inference_batch_normalization_143_layer_call_fn_211510

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_143_layer_call_and_return_conditional_losses_2094652
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§

S__inference_batch_normalization_146_layer_call_and_return_conditional_losses_212005

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


,__inference_conv_layer3_layer_call_fn_211903

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿGG*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_layer3_layer_call_and_return_conditional_losses_2102742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿGG2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿII@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@
 
_user_specified_nameinputs
	
¯
G__inference_conv_layer1_layer_call_and_return_conditional_losses_210072

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©© *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©© 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©© 2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©© 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ¬¬:::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
 
_user_specified_nameinputs
×^

I__inference_sequential_24_layer_call_and_return_conditional_losses_210729

inputs"
batch_normalization_143_210643"
batch_normalization_143_210645"
batch_normalization_143_210647"
batch_normalization_143_210649
conv_layer1_210652
conv_layer1_210654"
batch_normalization_144_210658"
batch_normalization_144_210660"
batch_normalization_144_210662"
batch_normalization_144_210664
conv_layer2_210667
conv_layer2_210669"
batch_normalization_145_210673"
batch_normalization_145_210675"
batch_normalization_145_210677"
batch_normalization_145_210679
conv_layer3_210682
conv_layer3_210684"
batch_normalization_146_210688"
batch_normalization_146_210690"
batch_normalization_146_210692"
batch_normalization_146_210694
conv_layer4_210697
conv_layer4_210699"
batch_normalization_147_210703"
batch_normalization_147_210705"
batch_normalization_147_210707"
batch_normalization_147_210709
dense_layer1_210713
dense_layer1_210715
dense_layer2_210718
dense_layer2_210720
output_layer_210723
output_layer_210725
identity¢$Dense_layer1/StatefulPartitionedCall¢$Dense_layer2/StatefulPartitionedCall¢/batch_normalization_143/StatefulPartitionedCall¢/batch_normalization_144/StatefulPartitionedCall¢/batch_normalization_145/StatefulPartitionedCall¢/batch_normalization_146/StatefulPartitionedCall¢/batch_normalization_147/StatefulPartitionedCall¢#conv_layer1/StatefulPartitionedCall¢#conv_layer2/StatefulPartitionedCall¢#conv_layer3/StatefulPartitionedCall¢#conv_layer4/StatefulPartitionedCall¢$output_layer/StatefulPartitionedCall®
/batch_normalization_143/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_143_210643batch_normalization_143_210645batch_normalization_143_210647batch_normalization_143_210649*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_143_layer_call_and_return_conditional_losses_21000721
/batch_normalization_143/StatefulPartitionedCallâ
#conv_layer1/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_143/StatefulPartitionedCall:output:0conv_layer1_210652conv_layer1_210654*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©© *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_layer1_layer_call_and_return_conditional_losses_2100722%
#conv_layer1/StatefulPartitionedCall¢
!max_pooling2d_129/PartitionedCallPartitionedCall,conv_layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_129_layer_call_and_return_conditional_losses_2095132#
!max_pooling2d_129/PartitionedCallÒ
/batch_normalization_144/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_129/PartitionedCall:output:0batch_normalization_144_210658batch_normalization_144_210660batch_normalization_144_210662batch_normalization_144_210664*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_144_layer_call_and_return_conditional_losses_21010821
/batch_normalization_144/StatefulPartitionedCallâ
#conv_layer2/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_144/StatefulPartitionedCall:output:0conv_layer2_210667conv_layer2_210669*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_layer2_layer_call_and_return_conditional_losses_2101732%
#conv_layer2/StatefulPartitionedCall 
!max_pooling2d_130/PartitionedCallPartitionedCall,conv_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_130_layer_call_and_return_conditional_losses_2096292#
!max_pooling2d_130/PartitionedCallÐ
/batch_normalization_145/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_130/PartitionedCall:output:0batch_normalization_145_210673batch_normalization_145_210675batch_normalization_145_210677batch_normalization_145_210679*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_145_layer_call_and_return_conditional_losses_21020921
/batch_normalization_145/StatefulPartitionedCallá
#conv_layer3/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_145/StatefulPartitionedCall:output:0conv_layer3_210682conv_layer3_210684*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿGG*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_layer3_layer_call_and_return_conditional_losses_2102742%
#conv_layer3/StatefulPartitionedCall¡
!max_pooling2d_131/PartitionedCallPartitionedCall,conv_layer3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_131_layer_call_and_return_conditional_losses_2097452#
!max_pooling2d_131/PartitionedCallÑ
/batch_normalization_146/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_131/PartitionedCall:output:0batch_normalization_146_210688batch_normalization_146_210690batch_normalization_146_210692batch_normalization_146_210694*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_146_layer_call_and_return_conditional_losses_21031021
/batch_normalization_146/StatefulPartitionedCallá
#conv_layer4/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_146/StatefulPartitionedCall:output:0conv_layer4_210697conv_layer4_210699*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_layer4_layer_call_and_return_conditional_losses_2103752%
#conv_layer4/StatefulPartitionedCall¡
!max_pooling2d_132/PartitionedCallPartitionedCall,conv_layer4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_132_layer_call_and_return_conditional_losses_2098612#
!max_pooling2d_132/PartitionedCallÑ
/batch_normalization_147/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_132/PartitionedCall:output:0batch_normalization_147_210703batch_normalization_147_210705batch_normalization_147_210707batch_normalization_147_210709*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_147_layer_call_and_return_conditional_losses_21041121
/batch_normalization_147/StatefulPartitionedCallÃ
+global_average_pooling2d_26/PartitionedCallPartitionedCall8batch_normalization_147/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling2d_26_layer_call_and_return_conditional_losses_2099782-
+global_average_pooling2d_26/PartitionedCallÙ
$Dense_layer1/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling2d_26/PartitionedCall:output:0dense_layer1_210713dense_layer1_210715*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Dense_layer1_layer_call_and_return_conditional_losses_2104772&
$Dense_layer1/StatefulPartitionedCallÒ
$Dense_layer2/StatefulPartitionedCallStatefulPartitionedCall-Dense_layer1/StatefulPartitionedCall:output:0dense_layer2_210718dense_layer2_210720*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Dense_layer2_layer_call_and_return_conditional_losses_2105042&
$Dense_layer2/StatefulPartitionedCallÒ
$output_layer/StatefulPartitionedCallStatefulPartitionedCall-Dense_layer2/StatefulPartitionedCall:output:0output_layer_210723output_layer_210725*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_2105312&
$output_layer/StatefulPartitionedCall
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0%^Dense_layer1/StatefulPartitionedCall%^Dense_layer2/StatefulPartitionedCall0^batch_normalization_143/StatefulPartitionedCall0^batch_normalization_144/StatefulPartitionedCall0^batch_normalization_145/StatefulPartitionedCall0^batch_normalization_146/StatefulPartitionedCall0^batch_normalization_147/StatefulPartitionedCall$^conv_layer1/StatefulPartitionedCall$^conv_layer2/StatefulPartitionedCall$^conv_layer3/StatefulPartitionedCall$^conv_layer4/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*º
_input_shapes¨
¥:ÿÿÿÿÿÿÿÿÿ¬¬::::::::::::::::::::::::::::::::::2L
$Dense_layer1/StatefulPartitionedCall$Dense_layer1/StatefulPartitionedCall2L
$Dense_layer2/StatefulPartitionedCall$Dense_layer2/StatefulPartitionedCall2b
/batch_normalization_143/StatefulPartitionedCall/batch_normalization_143/StatefulPartitionedCall2b
/batch_normalization_144/StatefulPartitionedCall/batch_normalization_144/StatefulPartitionedCall2b
/batch_normalization_145/StatefulPartitionedCall/batch_normalization_145/StatefulPartitionedCall2b
/batch_normalization_146/StatefulPartitionedCall/batch_normalization_146/StatefulPartitionedCall2b
/batch_normalization_147/StatefulPartitionedCall/batch_normalization_147/StatefulPartitionedCall2J
#conv_layer1/StatefulPartitionedCall#conv_layer1/StatefulPartitionedCall2J
#conv_layer2/StatefulPartitionedCall#conv_layer2/StatefulPartitionedCall2J
#conv_layer3/StatefulPartitionedCall#conv_layer3/StatefulPartitionedCall2J
#conv_layer4/StatefulPartitionedCall#conv_layer4/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
 
_user_specified_nameinputs
Ë
°
S__inference_batch_normalization_144_layer_call_and_return_conditional_losses_211627

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
´
ÿ
.__inference_sequential_24_layer_call_fn_210800
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32
identity¢StatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*:
_read_only_resource_inputs
 !"*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_24_layer_call_and_return_conditional_losses_2107292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*º
_input_shapes¨
¥:ÿÿÿÿÿÿÿÿÿ¬¬::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
%
_user_specified_nameinput_layer
­
°
H__inference_Dense_layer2_layer_call_and_return_conditional_losses_210504

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
Ò

S__inference_batch_normalization_145_layer_call_and_return_conditional_losses_211857

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿII@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿII@:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@
 
_user_specified_nameinputs
¾
ÿ
.__inference_sequential_24_layer_call_fn_210962
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32
identity¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_24_layer_call_and_return_conditional_losses_2108912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*º
_input_shapes¨
¥:ÿÿÿÿÿÿÿÿÿ¬¬::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
%
_user_specified_nameinput_layer

°
S__inference_batch_normalization_145_layer_call_and_return_conditional_losses_211839

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿII@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿII@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@
 
_user_specified_nameinputs
­
°
H__inference_Dense_layer2_layer_call_and_return_conditional_losses_212210

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs

õ
$__inference_signature_wrapper_211045
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_2094032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*º
_input_shapes¨
¥:ÿÿÿÿÿÿÿÿÿ¬¬::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
%
_user_specified_nameinput_layer

i
M__inference_max_pooling2d_130_layer_call_and_return_conditional_losses_209629

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

°
S__inference_batch_normalization_147_layer_call_and_return_conditional_losses_212135

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

°
S__inference_batch_normalization_143_layer_call_and_return_conditional_losses_211543

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ú
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ¬¬:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ¬¬::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
 
_user_specified_nameinputs
Ë
°
S__inference_batch_normalization_143_layer_call_and_return_conditional_losses_211479

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

°
S__inference_batch_normalization_144_layer_call_and_return_conditional_losses_211691

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ú
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
×
°
S__inference_batch_normalization_146_layer_call_and_return_conditional_losses_209813

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


,__inference_conv_layer1_layer_call_fn_211607

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©© *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_layer1_layer_call_and_return_conditional_losses_2100722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©© 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ¬¬::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
 
_user_specified_nameinputs
Þ

S__inference_batch_normalization_146_layer_call_and_return_conditional_losses_211941

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ##:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ##:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##
 
_user_specified_nameinputs
æ
«
8__inference_batch_normalization_144_layer_call_fn_211735

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_144_layer_call_and_return_conditional_losses_2101262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	
¯
G__inference_conv_layer4_layer_call_and_return_conditional_losses_210375

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!!*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!!2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!!2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!!2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ##:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##
 
_user_specified_nameinputs
Ú

S__inference_batch_normalization_143_layer_call_and_return_conditional_losses_210025

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ¬¬:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3r
IdentityIdentityFusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ¬¬:::::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
 
_user_specified_nameinputs
¬²

I__inference_sequential_24_layer_call_and_return_conditional_losses_211184

inputs3
/batch_normalization_143_readvariableop_resource5
1batch_normalization_143_readvariableop_1_resourceD
@batch_normalization_143_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_143_fusedbatchnormv3_readvariableop_1_resource.
*conv_layer1_conv2d_readvariableop_resource/
+conv_layer1_biasadd_readvariableop_resource3
/batch_normalization_144_readvariableop_resource5
1batch_normalization_144_readvariableop_1_resourceD
@batch_normalization_144_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_144_fusedbatchnormv3_readvariableop_1_resource.
*conv_layer2_conv2d_readvariableop_resource/
+conv_layer2_biasadd_readvariableop_resource3
/batch_normalization_145_readvariableop_resource5
1batch_normalization_145_readvariableop_1_resourceD
@batch_normalization_145_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_145_fusedbatchnormv3_readvariableop_1_resource.
*conv_layer3_conv2d_readvariableop_resource/
+conv_layer3_biasadd_readvariableop_resource3
/batch_normalization_146_readvariableop_resource5
1batch_normalization_146_readvariableop_1_resourceD
@batch_normalization_146_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_146_fusedbatchnormv3_readvariableop_1_resource.
*conv_layer4_conv2d_readvariableop_resource/
+conv_layer4_biasadd_readvariableop_resource3
/batch_normalization_147_readvariableop_resource5
1batch_normalization_147_readvariableop_1_resourceD
@batch_normalization_147_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_147_fusedbatchnormv3_readvariableop_1_resource/
+dense_layer1_matmul_readvariableop_resource0
,dense_layer1_biasadd_readvariableop_resource/
+dense_layer2_matmul_readvariableop_resource0
,dense_layer2_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identity¢&batch_normalization_143/AssignNewValue¢(batch_normalization_143/AssignNewValue_1¢&batch_normalization_144/AssignNewValue¢(batch_normalization_144/AssignNewValue_1¢&batch_normalization_145/AssignNewValue¢(batch_normalization_145/AssignNewValue_1¢&batch_normalization_146/AssignNewValue¢(batch_normalization_146/AssignNewValue_1¢&batch_normalization_147/AssignNewValue¢(batch_normalization_147/AssignNewValue_1¼
&batch_normalization_143/ReadVariableOpReadVariableOp/batch_normalization_143_readvariableop_resource*
_output_shapes
:*
dtype02(
&batch_normalization_143/ReadVariableOpÂ
(batch_normalization_143/ReadVariableOp_1ReadVariableOp1batch_normalization_143_readvariableop_1_resource*
_output_shapes
:*
dtype02*
(batch_normalization_143/ReadVariableOp_1ï
7batch_normalization_143/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_143_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_143/FusedBatchNormV3/ReadVariableOpõ
9batch_normalization_143/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_143_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02;
9batch_normalization_143/FusedBatchNormV3/ReadVariableOp_1ê
(batch_normalization_143/FusedBatchNormV3FusedBatchNormV3inputs.batch_normalization_143/ReadVariableOp:value:00batch_normalization_143/ReadVariableOp_1:value:0?batch_normalization_143/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_143/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ¬¬:::::*
epsilon%o:*
exponential_avg_factor%
×#<2*
(batch_normalization_143/FusedBatchNormV3
&batch_normalization_143/AssignNewValueAssignVariableOp@batch_normalization_143_fusedbatchnormv3_readvariableop_resource5batch_normalization_143/FusedBatchNormV3:batch_mean:08^batch_normalization_143/FusedBatchNormV3/ReadVariableOp*S
_classI
GEloc:@batch_normalization_143/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02(
&batch_normalization_143/AssignNewValue
(batch_normalization_143/AssignNewValue_1AssignVariableOpBbatch_normalization_143_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_143/FusedBatchNormV3:batch_variance:0:^batch_normalization_143/FusedBatchNormV3/ReadVariableOp_1*U
_classK
IGloc:@batch_normalization_143/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02*
(batch_normalization_143/AssignNewValue_1¹
!conv_layer1/Conv2D/ReadVariableOpReadVariableOp*conv_layer1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02#
!conv_layer1/Conv2D/ReadVariableOpð
conv_layer1/Conv2DConv2D,batch_normalization_143/FusedBatchNormV3:y:0)conv_layer1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©© *
paddingVALID*
strides
2
conv_layer1/Conv2D°
"conv_layer1/BiasAdd/ReadVariableOpReadVariableOp+conv_layer1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"conv_layer1/BiasAdd/ReadVariableOpº
conv_layer1/BiasAddBiasAddconv_layer1/Conv2D:output:0*conv_layer1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©© 2
conv_layer1/BiasAdd
conv_layer1/ReluReluconv_layer1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©© 2
conv_layer1/ReluÐ
max_pooling2d_129/MaxPoolMaxPoolconv_layer1/Relu:activations:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_129/MaxPool¼
&batch_normalization_144/ReadVariableOpReadVariableOp/batch_normalization_144_readvariableop_resource*
_output_shapes
: *
dtype02(
&batch_normalization_144/ReadVariableOpÂ
(batch_normalization_144/ReadVariableOp_1ReadVariableOp1batch_normalization_144_readvariableop_1_resource*
_output_shapes
: *
dtype02*
(batch_normalization_144/ReadVariableOp_1ï
7batch_normalization_144/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_144_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype029
7batch_normalization_144/FusedBatchNormV3/ReadVariableOpõ
9batch_normalization_144/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_144_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02;
9batch_normalization_144/FusedBatchNormV3/ReadVariableOp_1
(batch_normalization_144/FusedBatchNormV3FusedBatchNormV3"max_pooling2d_129/MaxPool:output:0.batch_normalization_144/ReadVariableOp:value:00batch_normalization_144/ReadVariableOp_1:value:0?batch_normalization_144/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_144/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2*
(batch_normalization_144/FusedBatchNormV3
&batch_normalization_144/AssignNewValueAssignVariableOp@batch_normalization_144_fusedbatchnormv3_readvariableop_resource5batch_normalization_144/FusedBatchNormV3:batch_mean:08^batch_normalization_144/FusedBatchNormV3/ReadVariableOp*S
_classI
GEloc:@batch_normalization_144/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02(
&batch_normalization_144/AssignNewValue
(batch_normalization_144/AssignNewValue_1AssignVariableOpBbatch_normalization_144_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_144/FusedBatchNormV3:batch_variance:0:^batch_normalization_144/FusedBatchNormV3/ReadVariableOp_1*U
_classK
IGloc:@batch_normalization_144/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02*
(batch_normalization_144/AssignNewValue_1¹
!conv_layer2/Conv2D/ReadVariableOpReadVariableOp*conv_layer2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02#
!conv_layer2/Conv2D/ReadVariableOpð
conv_layer2/Conv2DConv2D,batch_normalization_144/FusedBatchNormV3:y:0)conv_layer2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv_layer2/Conv2D°
"conv_layer2/BiasAdd/ReadVariableOpReadVariableOp+conv_layer2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"conv_layer2/BiasAdd/ReadVariableOpº
conv_layer2/BiasAddBiasAddconv_layer2/Conv2D:output:0*conv_layer2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv_layer2/BiasAdd
conv_layer2/ReluReluconv_layer2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv_layer2/ReluÎ
max_pooling2d_130/MaxPoolMaxPoolconv_layer2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_130/MaxPool¼
&batch_normalization_145/ReadVariableOpReadVariableOp/batch_normalization_145_readvariableop_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_145/ReadVariableOpÂ
(batch_normalization_145/ReadVariableOp_1ReadVariableOp1batch_normalization_145_readvariableop_1_resource*
_output_shapes
:@*
dtype02*
(batch_normalization_145/ReadVariableOp_1ï
7batch_normalization_145/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_145_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype029
7batch_normalization_145/FusedBatchNormV3/ReadVariableOpõ
9batch_normalization_145/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_145_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02;
9batch_normalization_145/FusedBatchNormV3/ReadVariableOp_1
(batch_normalization_145/FusedBatchNormV3FusedBatchNormV3"max_pooling2d_130/MaxPool:output:0.batch_normalization_145/ReadVariableOp:value:00batch_normalization_145/ReadVariableOp_1:value:0?batch_normalization_145/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_145/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿII@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2*
(batch_normalization_145/FusedBatchNormV3
&batch_normalization_145/AssignNewValueAssignVariableOp@batch_normalization_145_fusedbatchnormv3_readvariableop_resource5batch_normalization_145/FusedBatchNormV3:batch_mean:08^batch_normalization_145/FusedBatchNormV3/ReadVariableOp*S
_classI
GEloc:@batch_normalization_145/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02(
&batch_normalization_145/AssignNewValue
(batch_normalization_145/AssignNewValue_1AssignVariableOpBbatch_normalization_145_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_145/FusedBatchNormV3:batch_variance:0:^batch_normalization_145/FusedBatchNormV3/ReadVariableOp_1*U
_classK
IGloc:@batch_normalization_145/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02*
(batch_normalization_145/AssignNewValue_1º
!conv_layer3/Conv2D/ReadVariableOpReadVariableOp*conv_layer3_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02#
!conv_layer3/Conv2D/ReadVariableOpï
conv_layer3/Conv2DConv2D,batch_normalization_145/FusedBatchNormV3:y:0)conv_layer3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿGG*
paddingVALID*
strides
2
conv_layer3/Conv2D±
"conv_layer3/BiasAdd/ReadVariableOpReadVariableOp+conv_layer3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"conv_layer3/BiasAdd/ReadVariableOp¹
conv_layer3/BiasAddBiasAddconv_layer3/Conv2D:output:0*conv_layer3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿGG2
conv_layer3/BiasAdd
conv_layer3/ReluReluconv_layer3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿGG2
conv_layer3/ReluÏ
max_pooling2d_131/MaxPoolMaxPoolconv_layer3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
ksize
*
paddingVALID*
strides
2
max_pooling2d_131/MaxPool½
&batch_normalization_146/ReadVariableOpReadVariableOp/batch_normalization_146_readvariableop_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_146/ReadVariableOpÃ
(batch_normalization_146/ReadVariableOp_1ReadVariableOp1batch_normalization_146_readvariableop_1_resource*
_output_shapes	
:*
dtype02*
(batch_normalization_146/ReadVariableOp_1ð
7batch_normalization_146/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_146_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype029
7batch_normalization_146/FusedBatchNormV3/ReadVariableOpö
9batch_normalization_146/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_146_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02;
9batch_normalization_146/FusedBatchNormV3/ReadVariableOp_1
(batch_normalization_146/FusedBatchNormV3FusedBatchNormV3"max_pooling2d_131/MaxPool:output:0.batch_normalization_146/ReadVariableOp:value:00batch_normalization_146/ReadVariableOp_1:value:0?batch_normalization_146/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_146/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ##:::::*
epsilon%o:*
exponential_avg_factor%
×#<2*
(batch_normalization_146/FusedBatchNormV3
&batch_normalization_146/AssignNewValueAssignVariableOp@batch_normalization_146_fusedbatchnormv3_readvariableop_resource5batch_normalization_146/FusedBatchNormV3:batch_mean:08^batch_normalization_146/FusedBatchNormV3/ReadVariableOp*S
_classI
GEloc:@batch_normalization_146/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02(
&batch_normalization_146/AssignNewValue
(batch_normalization_146/AssignNewValue_1AssignVariableOpBbatch_normalization_146_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_146/FusedBatchNormV3:batch_variance:0:^batch_normalization_146/FusedBatchNormV3/ReadVariableOp_1*U
_classK
IGloc:@batch_normalization_146/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02*
(batch_normalization_146/AssignNewValue_1»
!conv_layer4/Conv2D/ReadVariableOpReadVariableOp*conv_layer4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02#
!conv_layer4/Conv2D/ReadVariableOpï
conv_layer4/Conv2DConv2D,batch_normalization_146/FusedBatchNormV3:y:0)conv_layer4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!!*
paddingVALID*
strides
2
conv_layer4/Conv2D±
"conv_layer4/BiasAdd/ReadVariableOpReadVariableOp+conv_layer4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"conv_layer4/BiasAdd/ReadVariableOp¹
conv_layer4/BiasAddBiasAddconv_layer4/Conv2D:output:0*conv_layer4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!!2
conv_layer4/BiasAdd
conv_layer4/ReluReluconv_layer4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!!2
conv_layer4/ReluÏ
max_pooling2d_132/MaxPoolMaxPoolconv_layer4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_132/MaxPool½
&batch_normalization_147/ReadVariableOpReadVariableOp/batch_normalization_147_readvariableop_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_147/ReadVariableOpÃ
(batch_normalization_147/ReadVariableOp_1ReadVariableOp1batch_normalization_147_readvariableop_1_resource*
_output_shapes	
:*
dtype02*
(batch_normalization_147/ReadVariableOp_1ð
7batch_normalization_147/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_147_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype029
7batch_normalization_147/FusedBatchNormV3/ReadVariableOpö
9batch_normalization_147/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_147_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02;
9batch_normalization_147/FusedBatchNormV3/ReadVariableOp_1
(batch_normalization_147/FusedBatchNormV3FusedBatchNormV3"max_pooling2d_132/MaxPool:output:0.batch_normalization_147/ReadVariableOp:value:00batch_normalization_147/ReadVariableOp_1:value:0?batch_normalization_147/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_147/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2*
(batch_normalization_147/FusedBatchNormV3
&batch_normalization_147/AssignNewValueAssignVariableOp@batch_normalization_147_fusedbatchnormv3_readvariableop_resource5batch_normalization_147/FusedBatchNormV3:batch_mean:08^batch_normalization_147/FusedBatchNormV3/ReadVariableOp*S
_classI
GEloc:@batch_normalization_147/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02(
&batch_normalization_147/AssignNewValue
(batch_normalization_147/AssignNewValue_1AssignVariableOpBbatch_normalization_147_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_147/FusedBatchNormV3:batch_variance:0:^batch_normalization_147/FusedBatchNormV3/ReadVariableOp_1*U
_classK
IGloc:@batch_normalization_147/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02*
(batch_normalization_147/AssignNewValue_1¹
2global_average_pooling2d_26/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      24
2global_average_pooling2d_26/Mean/reduction_indicesê
 global_average_pooling2d_26/MeanMean,batch_normalization_147/FusedBatchNormV3:y:0;global_average_pooling2d_26/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 global_average_pooling2d_26/Meanµ
"Dense_layer1/MatMul/ReadVariableOpReadVariableOp+dense_layer1_matmul_readvariableop_resource*
_output_shapes
:	2*
dtype02$
"Dense_layer1/MatMul/ReadVariableOp½
Dense_layer1/MatMulMatMul)global_average_pooling2d_26/Mean:output:0*Dense_layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
Dense_layer1/MatMul³
#Dense_layer1/BiasAdd/ReadVariableOpReadVariableOp,dense_layer1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02%
#Dense_layer1/BiasAdd/ReadVariableOpµ
Dense_layer1/BiasAddBiasAddDense_layer1/MatMul:product:0+Dense_layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
Dense_layer1/BiasAdd
Dense_layer1/ReluReluDense_layer1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
Dense_layer1/Relu´
"Dense_layer2/MatMul/ReadVariableOpReadVariableOp+dense_layer2_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02$
"Dense_layer2/MatMul/ReadVariableOp³
Dense_layer2/MatMulMatMulDense_layer1/Relu:activations:0*Dense_layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Dense_layer2/MatMul³
#Dense_layer2/BiasAdd/ReadVariableOpReadVariableOp,dense_layer2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02%
#Dense_layer2/BiasAdd/ReadVariableOpµ
Dense_layer2/BiasAddBiasAddDense_layer2/MatMul:product:0+Dense_layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Dense_layer2/BiasAdd
Dense_layer2/ReluReluDense_layer2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Dense_layer2/Relu´
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02$
"output_layer/MatMul/ReadVariableOp³
output_layer/MatMulMatMulDense_layer2/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output_layer/MatMul³
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#output_layer/BiasAdd/ReadVariableOpµ
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output_layer/BiasAdd
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output_layer/Softmax
IdentityIdentityoutput_layer/Softmax:softmax:0'^batch_normalization_143/AssignNewValue)^batch_normalization_143/AssignNewValue_1'^batch_normalization_144/AssignNewValue)^batch_normalization_144/AssignNewValue_1'^batch_normalization_145/AssignNewValue)^batch_normalization_145/AssignNewValue_1'^batch_normalization_146/AssignNewValue)^batch_normalization_146/AssignNewValue_1'^batch_normalization_147/AssignNewValue)^batch_normalization_147/AssignNewValue_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*º
_input_shapes¨
¥:ÿÿÿÿÿÿÿÿÿ¬¬::::::::::::::::::::::::::::::::::2P
&batch_normalization_143/AssignNewValue&batch_normalization_143/AssignNewValue2T
(batch_normalization_143/AssignNewValue_1(batch_normalization_143/AssignNewValue_12P
&batch_normalization_144/AssignNewValue&batch_normalization_144/AssignNewValue2T
(batch_normalization_144/AssignNewValue_1(batch_normalization_144/AssignNewValue_12P
&batch_normalization_145/AssignNewValue&batch_normalization_145/AssignNewValue2T
(batch_normalization_145/AssignNewValue_1(batch_normalization_145/AssignNewValue_12P
&batch_normalization_146/AssignNewValue&batch_normalization_146/AssignNewValue2T
(batch_normalization_146/AssignNewValue_1(batch_normalization_146/AssignNewValue_12P
&batch_normalization_147/AssignNewValue&batch_normalization_147/AssignNewValue2T
(batch_normalization_147/AssignNewValue_1(batch_normalization_147/AssignNewValue_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
 
_user_specified_nameinputs
á^

I__inference_sequential_24_layer_call_and_return_conditional_losses_210891

inputs"
batch_normalization_143_210805"
batch_normalization_143_210807"
batch_normalization_143_210809"
batch_normalization_143_210811
conv_layer1_210814
conv_layer1_210816"
batch_normalization_144_210820"
batch_normalization_144_210822"
batch_normalization_144_210824"
batch_normalization_144_210826
conv_layer2_210829
conv_layer2_210831"
batch_normalization_145_210835"
batch_normalization_145_210837"
batch_normalization_145_210839"
batch_normalization_145_210841
conv_layer3_210844
conv_layer3_210846"
batch_normalization_146_210850"
batch_normalization_146_210852"
batch_normalization_146_210854"
batch_normalization_146_210856
conv_layer4_210859
conv_layer4_210861"
batch_normalization_147_210865"
batch_normalization_147_210867"
batch_normalization_147_210869"
batch_normalization_147_210871
dense_layer1_210875
dense_layer1_210877
dense_layer2_210880
dense_layer2_210882
output_layer_210885
output_layer_210887
identity¢$Dense_layer1/StatefulPartitionedCall¢$Dense_layer2/StatefulPartitionedCall¢/batch_normalization_143/StatefulPartitionedCall¢/batch_normalization_144/StatefulPartitionedCall¢/batch_normalization_145/StatefulPartitionedCall¢/batch_normalization_146/StatefulPartitionedCall¢/batch_normalization_147/StatefulPartitionedCall¢#conv_layer1/StatefulPartitionedCall¢#conv_layer2/StatefulPartitionedCall¢#conv_layer3/StatefulPartitionedCall¢#conv_layer4/StatefulPartitionedCall¢$output_layer/StatefulPartitionedCall°
/batch_normalization_143/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_143_210805batch_normalization_143_210807batch_normalization_143_210809batch_normalization_143_210811*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_143_layer_call_and_return_conditional_losses_21002521
/batch_normalization_143/StatefulPartitionedCallâ
#conv_layer1/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_143/StatefulPartitionedCall:output:0conv_layer1_210814conv_layer1_210816*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©© *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_layer1_layer_call_and_return_conditional_losses_2100722%
#conv_layer1/StatefulPartitionedCall¢
!max_pooling2d_129/PartitionedCallPartitionedCall,conv_layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_129_layer_call_and_return_conditional_losses_2095132#
!max_pooling2d_129/PartitionedCallÔ
/batch_normalization_144/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_129/PartitionedCall:output:0batch_normalization_144_210820batch_normalization_144_210822batch_normalization_144_210824batch_normalization_144_210826*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_144_layer_call_and_return_conditional_losses_21012621
/batch_normalization_144/StatefulPartitionedCallâ
#conv_layer2/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_144/StatefulPartitionedCall:output:0conv_layer2_210829conv_layer2_210831*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_layer2_layer_call_and_return_conditional_losses_2101732%
#conv_layer2/StatefulPartitionedCall 
!max_pooling2d_130/PartitionedCallPartitionedCall,conv_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_130_layer_call_and_return_conditional_losses_2096292#
!max_pooling2d_130/PartitionedCallÒ
/batch_normalization_145/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_130/PartitionedCall:output:0batch_normalization_145_210835batch_normalization_145_210837batch_normalization_145_210839batch_normalization_145_210841*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿII@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_145_layer_call_and_return_conditional_losses_21022721
/batch_normalization_145/StatefulPartitionedCallá
#conv_layer3/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_145/StatefulPartitionedCall:output:0conv_layer3_210844conv_layer3_210846*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿGG*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_layer3_layer_call_and_return_conditional_losses_2102742%
#conv_layer3/StatefulPartitionedCall¡
!max_pooling2d_131/PartitionedCallPartitionedCall,conv_layer3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_131_layer_call_and_return_conditional_losses_2097452#
!max_pooling2d_131/PartitionedCallÓ
/batch_normalization_146/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_131/PartitionedCall:output:0batch_normalization_146_210850batch_normalization_146_210852batch_normalization_146_210854batch_normalization_146_210856*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_146_layer_call_and_return_conditional_losses_21032821
/batch_normalization_146/StatefulPartitionedCallá
#conv_layer4/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_146/StatefulPartitionedCall:output:0conv_layer4_210859conv_layer4_210861*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ!!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv_layer4_layer_call_and_return_conditional_losses_2103752%
#conv_layer4/StatefulPartitionedCall¡
!max_pooling2d_132/PartitionedCallPartitionedCall,conv_layer4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_132_layer_call_and_return_conditional_losses_2098612#
!max_pooling2d_132/PartitionedCallÓ
/batch_normalization_147/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_132/PartitionedCall:output:0batch_normalization_147_210865batch_normalization_147_210867batch_normalization_147_210869batch_normalization_147_210871*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_147_layer_call_and_return_conditional_losses_21042921
/batch_normalization_147/StatefulPartitionedCallÃ
+global_average_pooling2d_26/PartitionedCallPartitionedCall8batch_normalization_147/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling2d_26_layer_call_and_return_conditional_losses_2099782-
+global_average_pooling2d_26/PartitionedCallÙ
$Dense_layer1/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling2d_26/PartitionedCall:output:0dense_layer1_210875dense_layer1_210877*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Dense_layer1_layer_call_and_return_conditional_losses_2104772&
$Dense_layer1/StatefulPartitionedCallÒ
$Dense_layer2/StatefulPartitionedCallStatefulPartitionedCall-Dense_layer1/StatefulPartitionedCall:output:0dense_layer2_210880dense_layer2_210882*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_Dense_layer2_layer_call_and_return_conditional_losses_2105042&
$Dense_layer2/StatefulPartitionedCallÒ
$output_layer/StatefulPartitionedCallStatefulPartitionedCall-Dense_layer2/StatefulPartitionedCall:output:0output_layer_210885output_layer_210887*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_2105312&
$output_layer/StatefulPartitionedCall
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0%^Dense_layer1/StatefulPartitionedCall%^Dense_layer2/StatefulPartitionedCall0^batch_normalization_143/StatefulPartitionedCall0^batch_normalization_144/StatefulPartitionedCall0^batch_normalization_145/StatefulPartitionedCall0^batch_normalization_146/StatefulPartitionedCall0^batch_normalization_147/StatefulPartitionedCall$^conv_layer1/StatefulPartitionedCall$^conv_layer2/StatefulPartitionedCall$^conv_layer3/StatefulPartitionedCall$^conv_layer4/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*º
_input_shapes¨
¥:ÿÿÿÿÿÿÿÿÿ¬¬::::::::::::::::::::::::::::::::::2L
$Dense_layer1/StatefulPartitionedCall$Dense_layer1/StatefulPartitionedCall2L
$Dense_layer2/StatefulPartitionedCall$Dense_layer2/StatefulPartitionedCall2b
/batch_normalization_143/StatefulPartitionedCall/batch_normalization_143/StatefulPartitionedCall2b
/batch_normalization_144/StatefulPartitionedCall/batch_normalization_144/StatefulPartitionedCall2b
/batch_normalization_145/StatefulPartitionedCall/batch_normalization_145/StatefulPartitionedCall2b
/batch_normalization_146/StatefulPartitionedCall/batch_normalization_146/StatefulPartitionedCall2b
/batch_normalization_147/StatefulPartitionedCall/batch_normalization_147/StatefulPartitionedCall2J
#conv_layer1/StatefulPartitionedCall#conv_layer1/StatefulPartitionedCall2J
#conv_layer2/StatefulPartitionedCall#conv_layer2/StatefulPartitionedCall2J
#conv_layer3/StatefulPartitionedCall#conv_layer3/StatefulPartitionedCall2J
#conv_layer4/StatefulPartitionedCall#conv_layer4/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¬
 
_user_specified_nameinputs
Þ

S__inference_batch_normalization_147_layer_call_and_return_conditional_losses_210429

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
«
8__inference_batch_normalization_147_layer_call_fn_212115

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_147_layer_call_and_return_conditional_losses_2099602
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

°
S__inference_batch_normalization_147_layer_call_and_return_conditional_losses_210411

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Á
serving_default­
M
input_layer>
serving_default_input_layer:0ÿÿÿÿÿÿÿÿÿ¬¬@
output_layer0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:úó
ª
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer-13
layer_with_weights-9
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
__call__
_default_save_signature
+&call_and_return_all_conditional_losses"½
_tf_keras_sequential{"class_name": "Sequential", "name": "sequential_24", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_24", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 300, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_143", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv_layer1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_129", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_144", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv_layer2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_130", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_145", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv_layer3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_131", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_146", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv_layer4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_132", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_147", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d_26", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "Dense_layer1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Dense_layer2", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300, 300, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_24", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 300, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_143", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv_layer1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_129", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_144", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv_layer2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_130", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_145", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv_layer3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_131", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_146", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv_layer4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_132", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_147", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d_26", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "Dense_layer1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Dense_layer2", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
À	
axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
 	keras_api
 __call__
+¡&call_and_return_all_conditional_losses"ê
_tf_keras_layerÐ{"class_name": "BatchNormalization", "name": "batch_normalization_143", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_143", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300, 300, 3]}}
û	

!kernel
"bias
#regularization_losses
$trainable_variables
%	variables
&	keras_api
¢__call__
+£&call_and_return_all_conditional_losses"Ô
_tf_keras_layerº{"class_name": "Conv2D", "name": "conv_layer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_layer1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300, 300, 3]}}

'regularization_losses
(trainable_variables
)	variables
*	keras_api
¤__call__
+¥&call_and_return_all_conditional_losses"ô
_tf_keras_layerÚ{"class_name": "MaxPooling2D", "name": "max_pooling2d_129", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_129", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Â	
+axis
	,gamma
-beta
.moving_mean
/moving_variance
0regularization_losses
1trainable_variables
2	variables
3	keras_api
¦__call__
+§&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "BatchNormalization", "name": "batch_normalization_144", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_144", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 148, 148, 32]}}
ý	

4kernel
5bias
6regularization_losses
7trainable_variables
8	variables
9	keras_api
¨__call__
+©&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Conv2D", "name": "conv_layer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_layer2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 148, 148, 32]}}

:regularization_losses
;trainable_variables
<	variables
=	keras_api
ª__call__
+«&call_and_return_all_conditional_losses"ô
_tf_keras_layerÚ{"class_name": "MaxPooling2D", "name": "max_pooling2d_130", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_130", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
À	
>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
¬__call__
+­&call_and_return_all_conditional_losses"ê
_tf_keras_layerÐ{"class_name": "BatchNormalization", "name": "batch_normalization_145", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_145", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 73, 73, 64]}}
ü	

Gkernel
Hbias
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
®__call__
+¯&call_and_return_all_conditional_losses"Õ
_tf_keras_layer»{"class_name": "Conv2D", "name": "conv_layer3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_layer3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 73, 73, 64]}}

Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
°__call__
+±&call_and_return_all_conditional_losses"ô
_tf_keras_layerÚ{"class_name": "MaxPooling2D", "name": "max_pooling2d_131", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_131", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Â	
Qaxis
	Rgamma
Sbeta
Tmoving_mean
Umoving_variance
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
²__call__
+³&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "BatchNormalization", "name": "batch_normalization_146", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_146", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 35, 128]}}
þ	

Zkernel
[bias
\regularization_losses
]trainable_variables
^	variables
_	keras_api
´__call__
+µ&call_and_return_all_conditional_losses"×
_tf_keras_layer½{"class_name": "Conv2D", "name": "conv_layer4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_layer4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 35, 128]}}

`regularization_losses
atrainable_variables
b	variables
c	keras_api
¶__call__
+·&call_and_return_all_conditional_losses"ô
_tf_keras_layerÚ{"class_name": "MaxPooling2D", "name": "max_pooling2d_132", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_132", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Â	
daxis
	egamma
fbeta
gmoving_mean
hmoving_variance
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
¸__call__
+¹&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "BatchNormalization", "name": "batch_normalization_147", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_147", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 256]}}

mregularization_losses
ntrainable_variables
o	variables
p	keras_api
º__call__
+»&call_and_return_all_conditional_losses"
_tf_keras_layerð{"class_name": "GlobalAveragePooling2D", "name": "global_average_pooling2d_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_average_pooling2d_26", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
þ

qkernel
rbias
sregularization_losses
ttrainable_variables
u	variables
v	keras_api
¼__call__
+½&call_and_return_all_conditional_losses"×
_tf_keras_layer½{"class_name": "Dense", "name": "Dense_layer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dense_layer1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
ý

wkernel
xbias
yregularization_losses
ztrainable_variables
{	variables
|	keras_api
¾__call__
+¿&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Dense", "name": "Dense_layer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dense_layer2", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}


}kernel
~bias
regularization_losses
trainable_variables
	variables
	keras_api
À__call__
+Á&call_and_return_all_conditional_losses"Ù
_tf_keras_layer¿{"class_name": "Dense", "name": "output_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
¸
	iter
beta_1
beta_2

decay
learning_ratemímî!mï"mð,mñ-mò4mó5mô?mõ@möGm÷HmøRmùSmúZmû[müemýfmþqmÿrmwmxm}m~mvv!v"v,v-v4v5v?v@vGvHvRvSvZv[vevfvqvrvwvxv}v~v"
	optimizer
 "
trackable_list_wrapper
Ö
0
1
!2
"3
,4
-5
46
57
?8
@9
G10
H11
R12
S13
Z14
[15
e16
f17
q18
r19
w20
x21
}22
~23"
trackable_list_wrapper
¦
0
1
2
3
!4
"5
,6
-7
.8
/9
410
511
?12
@13
A14
B15
G16
H17
R18
S19
T20
U21
Z22
[23
e24
f25
g26
h27
q28
r29
w30
x31
}32
~33"
trackable_list_wrapper
Ó
non_trainable_variables
metrics
regularization_losses
layer_metrics
layers
 layer_regularization_losses
trainable_variables
	variables
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
Âserving_default"
signature_map
 "
trackable_list_wrapper
+:)2batch_normalization_143/gamma
*:(2batch_normalization_143/beta
3:1 (2#batch_normalization_143/moving_mean
7:5 (2'batch_normalization_143/moving_variance
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
µ
non_trainable_variables
metrics
regularization_losses
layer_metrics
layers
 layer_regularization_losses
trainable_variables
	variables
 __call__
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses"
_generic_user_object
,:* 2conv_layer1/kernel
: 2conv_layer1/bias
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
µ
non_trainable_variables
metrics
#regularization_losses
layer_metrics
layers
 layer_regularization_losses
$trainable_variables
%	variables
¢__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
metrics
'regularization_losses
layer_metrics
layers
 layer_regularization_losses
(trainable_variables
)	variables
¤__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:) 2batch_normalization_144/gamma
*:( 2batch_normalization_144/beta
3:1  (2#batch_normalization_144/moving_mean
7:5  (2'batch_normalization_144/moving_variance
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
<
,0
-1
.2
/3"
trackable_list_wrapper
µ
non_trainable_variables
metrics
0regularization_losses
layer_metrics
layers
  layer_regularization_losses
1trainable_variables
2	variables
¦__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
,:* @2conv_layer2/kernel
:@2conv_layer2/bias
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
µ
¡non_trainable_variables
¢metrics
6regularization_losses
£layer_metrics
¤layers
 ¥layer_regularization_losses
7trainable_variables
8	variables
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¦non_trainable_variables
§metrics
:regularization_losses
¨layer_metrics
©layers
 ªlayer_regularization_losses
;trainable_variables
<	variables
ª__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)@2batch_normalization_145/gamma
*:(@2batch_normalization_145/beta
3:1@ (2#batch_normalization_145/moving_mean
7:5@ (2'batch_normalization_145/moving_variance
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
<
?0
@1
A2
B3"
trackable_list_wrapper
µ
«non_trainable_variables
¬metrics
Cregularization_losses
­layer_metrics
®layers
 ¯layer_regularization_losses
Dtrainable_variables
E	variables
¬__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
-:+@2conv_layer3/kernel
:2conv_layer3/bias
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
µ
°non_trainable_variables
±metrics
Iregularization_losses
²layer_metrics
³layers
 ´layer_regularization_losses
Jtrainable_variables
K	variables
®__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
µnon_trainable_variables
¶metrics
Mregularization_losses
·layer_metrics
¸layers
 ¹layer_regularization_losses
Ntrainable_variables
O	variables
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
,:*2batch_normalization_146/gamma
+:)2batch_normalization_146/beta
4:2 (2#batch_normalization_146/moving_mean
8:6 (2'batch_normalization_146/moving_variance
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
<
R0
S1
T2
U3"
trackable_list_wrapper
µ
ºnon_trainable_variables
»metrics
Vregularization_losses
¼layer_metrics
½layers
 ¾layer_regularization_losses
Wtrainable_variables
X	variables
²__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
.:,2conv_layer4/kernel
:2conv_layer4/bias
 "
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
µ
¿non_trainable_variables
Àmetrics
\regularization_losses
Álayer_metrics
Âlayers
 Ãlayer_regularization_losses
]trainable_variables
^	variables
´__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Änon_trainable_variables
Åmetrics
`regularization_losses
Ælayer_metrics
Çlayers
 Èlayer_regularization_losses
atrainable_variables
b	variables
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
,:*2batch_normalization_147/gamma
+:)2batch_normalization_147/beta
4:2 (2#batch_normalization_147/moving_mean
8:6 (2'batch_normalization_147/moving_variance
 "
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
<
e0
f1
g2
h3"
trackable_list_wrapper
µ
Énon_trainable_variables
Êmetrics
iregularization_losses
Ëlayer_metrics
Ìlayers
 Ílayer_regularization_losses
jtrainable_variables
k	variables
¸__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Înon_trainable_variables
Ïmetrics
mregularization_losses
Ðlayer_metrics
Ñlayers
 Òlayer_regularization_losses
ntrainable_variables
o	variables
º__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
&:$	22Dense_layer1/kernel
:22Dense_layer1/bias
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
µ
Ónon_trainable_variables
Ômetrics
sregularization_losses
Õlayer_metrics
Ölayers
 ×layer_regularization_losses
ttrainable_variables
u	variables
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
%:#2d2Dense_layer2/kernel
:d2Dense_layer2/bias
 "
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
µ
Ønon_trainable_variables
Ùmetrics
yregularization_losses
Úlayer_metrics
Ûlayers
 Ülayer_regularization_losses
ztrainable_variables
{	variables
¾__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
%:#d2output_layer/kernel
:2output_layer/bias
 "
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
·
Ýnon_trainable_variables
Þmetrics
regularization_losses
ßlayer_metrics
àlayers
 álayer_regularization_losses
trainable_variables
	variables
À__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
f
0
1
.2
/3
A4
B5
T6
U7
g8
h9"
trackable_list_wrapper
0
â0
ã1"
trackable_list_wrapper
 "
trackable_dict_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¿

ätotal

åcount
æ	variables
ç	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}


ètotal

écount
ê
_fn_kwargs
ë	variables
ì	keras_api"¸
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
ä0
å1"
trackable_list_wrapper
.
æ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
è0
é1"
trackable_list_wrapper
.
ë	variables"
_generic_user_object
0:.2$Adam/batch_normalization_143/gamma/m
/:-2#Adam/batch_normalization_143/beta/m
1:/ 2Adam/conv_layer1/kernel/m
#:! 2Adam/conv_layer1/bias/m
0:. 2$Adam/batch_normalization_144/gamma/m
/:- 2#Adam/batch_normalization_144/beta/m
1:/ @2Adam/conv_layer2/kernel/m
#:!@2Adam/conv_layer2/bias/m
0:.@2$Adam/batch_normalization_145/gamma/m
/:-@2#Adam/batch_normalization_145/beta/m
2:0@2Adam/conv_layer3/kernel/m
$:"2Adam/conv_layer3/bias/m
1:/2$Adam/batch_normalization_146/gamma/m
0:.2#Adam/batch_normalization_146/beta/m
3:12Adam/conv_layer4/kernel/m
$:"2Adam/conv_layer4/bias/m
1:/2$Adam/batch_normalization_147/gamma/m
0:.2#Adam/batch_normalization_147/beta/m
+:)	22Adam/Dense_layer1/kernel/m
$:"22Adam/Dense_layer1/bias/m
*:(2d2Adam/Dense_layer2/kernel/m
$:"d2Adam/Dense_layer2/bias/m
*:(d2Adam/output_layer/kernel/m
$:"2Adam/output_layer/bias/m
0:.2$Adam/batch_normalization_143/gamma/v
/:-2#Adam/batch_normalization_143/beta/v
1:/ 2Adam/conv_layer1/kernel/v
#:! 2Adam/conv_layer1/bias/v
0:. 2$Adam/batch_normalization_144/gamma/v
/:- 2#Adam/batch_normalization_144/beta/v
1:/ @2Adam/conv_layer2/kernel/v
#:!@2Adam/conv_layer2/bias/v
0:.@2$Adam/batch_normalization_145/gamma/v
/:-@2#Adam/batch_normalization_145/beta/v
2:0@2Adam/conv_layer3/kernel/v
$:"2Adam/conv_layer3/bias/v
1:/2$Adam/batch_normalization_146/gamma/v
0:.2#Adam/batch_normalization_146/beta/v
3:12Adam/conv_layer4/kernel/v
$:"2Adam/conv_layer4/bias/v
1:/2$Adam/batch_normalization_147/gamma/v
0:.2#Adam/batch_normalization_147/beta/v
+:)	22Adam/Dense_layer1/kernel/v
$:"22Adam/Dense_layer1/bias/v
*:(2d2Adam/Dense_layer2/kernel/v
$:"d2Adam/Dense_layer2/bias/v
*:(d2Adam/output_layer/kernel/v
$:"2Adam/output_layer/bias/v
2
.__inference_sequential_24_layer_call_fn_210800
.__inference_sequential_24_layer_call_fn_210962
.__inference_sequential_24_layer_call_fn_211386
.__inference_sequential_24_layer_call_fn_211459À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
í2ê
!__inference__wrapped_model_209403Ä
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *4¢1
/,
input_layerÿÿÿÿÿÿÿÿÿ¬¬
ò2ï
I__inference_sequential_24_layer_call_and_return_conditional_losses_211184
I__inference_sequential_24_layer_call_and_return_conditional_losses_210637
I__inference_sequential_24_layer_call_and_return_conditional_losses_210548
I__inference_sequential_24_layer_call_and_return_conditional_losses_211313À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¢2
8__inference_batch_normalization_143_layer_call_fn_211587
8__inference_batch_normalization_143_layer_call_fn_211510
8__inference_batch_normalization_143_layer_call_fn_211574
8__inference_batch_normalization_143_layer_call_fn_211523´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
S__inference_batch_normalization_143_layer_call_and_return_conditional_losses_211479
S__inference_batch_normalization_143_layer_call_and_return_conditional_losses_211497
S__inference_batch_normalization_143_layer_call_and_return_conditional_losses_211543
S__inference_batch_normalization_143_layer_call_and_return_conditional_losses_211561´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
,__inference_conv_layer1_layer_call_fn_211607¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_conv_layer1_layer_call_and_return_conditional_losses_211598¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
2__inference_max_pooling2d_129_layer_call_fn_209519à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
µ2²
M__inference_max_pooling2d_129_layer_call_and_return_conditional_losses_209513à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¢2
8__inference_batch_normalization_144_layer_call_fn_211722
8__inference_batch_normalization_144_layer_call_fn_211658
8__inference_batch_normalization_144_layer_call_fn_211671
8__inference_batch_normalization_144_layer_call_fn_211735´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
S__inference_batch_normalization_144_layer_call_and_return_conditional_losses_211691
S__inference_batch_normalization_144_layer_call_and_return_conditional_losses_211709
S__inference_batch_normalization_144_layer_call_and_return_conditional_losses_211627
S__inference_batch_normalization_144_layer_call_and_return_conditional_losses_211645´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
,__inference_conv_layer2_layer_call_fn_211755¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_conv_layer2_layer_call_and_return_conditional_losses_211746¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
2__inference_max_pooling2d_130_layer_call_fn_209635à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
µ2²
M__inference_max_pooling2d_130_layer_call_and_return_conditional_losses_209629à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¢2
8__inference_batch_normalization_145_layer_call_fn_211806
8__inference_batch_normalization_145_layer_call_fn_211870
8__inference_batch_normalization_145_layer_call_fn_211819
8__inference_batch_normalization_145_layer_call_fn_211883´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
S__inference_batch_normalization_145_layer_call_and_return_conditional_losses_211793
S__inference_batch_normalization_145_layer_call_and_return_conditional_losses_211839
S__inference_batch_normalization_145_layer_call_and_return_conditional_losses_211775
S__inference_batch_normalization_145_layer_call_and_return_conditional_losses_211857´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
,__inference_conv_layer3_layer_call_fn_211903¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_conv_layer3_layer_call_and_return_conditional_losses_211894¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
2__inference_max_pooling2d_131_layer_call_fn_209751à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
µ2²
M__inference_max_pooling2d_131_layer_call_and_return_conditional_losses_209745à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¢2
8__inference_batch_normalization_146_layer_call_fn_211967
8__inference_batch_normalization_146_layer_call_fn_211954
8__inference_batch_normalization_146_layer_call_fn_212031
8__inference_batch_normalization_146_layer_call_fn_212018´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
S__inference_batch_normalization_146_layer_call_and_return_conditional_losses_211923
S__inference_batch_normalization_146_layer_call_and_return_conditional_losses_212005
S__inference_batch_normalization_146_layer_call_and_return_conditional_losses_211941
S__inference_batch_normalization_146_layer_call_and_return_conditional_losses_211987´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
,__inference_conv_layer4_layer_call_fn_212051¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_conv_layer4_layer_call_and_return_conditional_losses_212042¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
2__inference_max_pooling2d_132_layer_call_fn_209867à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
µ2²
M__inference_max_pooling2d_132_layer_call_and_return_conditional_losses_209861à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¢2
8__inference_batch_normalization_147_layer_call_fn_212115
8__inference_batch_normalization_147_layer_call_fn_212179
8__inference_batch_normalization_147_layer_call_fn_212166
8__inference_batch_normalization_147_layer_call_fn_212102´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
S__inference_batch_normalization_147_layer_call_and_return_conditional_losses_212153
S__inference_batch_normalization_147_layer_call_and_return_conditional_losses_212089
S__inference_batch_normalization_147_layer_call_and_return_conditional_losses_212135
S__inference_batch_normalization_147_layer_call_and_return_conditional_losses_212071´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¤2¡
<__inference_global_average_pooling2d_26_layer_call_fn_209984à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¿2¼
W__inference_global_average_pooling2d_26_layer_call_and_return_conditional_losses_209978à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
×2Ô
-__inference_Dense_layer1_layer_call_fn_212199¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_Dense_layer1_layer_call_and_return_conditional_losses_212190¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_Dense_layer2_layer_call_fn_212219¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_Dense_layer2_layer_call_and_return_conditional_losses_212210¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_output_layer_layer_call_fn_212239¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_output_layer_layer_call_and_return_conditional_losses_212230¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
7B5
$__inference_signature_wrapper_211045input_layer©
H__inference_Dense_layer1_layer_call_and_return_conditional_losses_212190]qr0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ2
 
-__inference_Dense_layer1_layer_call_fn_212199Pqr0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ2¨
H__inference_Dense_layer2_layer_call_and_return_conditional_losses_212210\wx/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ2
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 
-__inference_Dense_layer2_layer_call_fn_212219Owx/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ2
ª "ÿÿÿÿÿÿÿÿÿdÇ
!__inference__wrapped_model_209403¡"!",-./45?@ABGHRSTUZ[efghqrwx}~>¢;
4¢1
/,
input_layerÿÿÿÿÿÿÿÿÿ¬¬
ª ";ª8
6
output_layer&#
output_layerÿÿÿÿÿÿÿÿÿî
S__inference_batch_normalization_143_layer_call_and_return_conditional_losses_211479M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 î
S__inference_batch_normalization_143_layer_call_and_return_conditional_losses_211497M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Í
S__inference_batch_normalization_143_layer_call_and_return_conditional_losses_211543v=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ¬¬
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ¬¬
 Í
S__inference_batch_normalization_143_layer_call_and_return_conditional_losses_211561v=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ¬¬
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ¬¬
 Æ
8__inference_batch_normalization_143_layer_call_fn_211510M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÆ
8__inference_batch_normalization_143_layer_call_fn_211523M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
8__inference_batch_normalization_143_layer_call_fn_211574i=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ¬¬
p
ª ""ÿÿÿÿÿÿÿÿÿ¬¬¥
8__inference_batch_normalization_143_layer_call_fn_211587i=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ¬¬
p 
ª ""ÿÿÿÿÿÿÿÿÿ¬¬î
S__inference_batch_normalization_144_layer_call_and_return_conditional_losses_211627,-./M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 î
S__inference_batch_normalization_144_layer_call_and_return_conditional_losses_211645,-./M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Í
S__inference_batch_normalization_144_layer_call_and_return_conditional_losses_211691v,-./=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 Í
S__inference_batch_normalization_144_layer_call_and_return_conditional_losses_211709v,-./=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 Æ
8__inference_batch_normalization_144_layer_call_fn_211658,-./M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Æ
8__inference_batch_normalization_144_layer_call_fn_211671,-./M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¥
8__inference_batch_normalization_144_layer_call_fn_211722i,-./=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ 
p
ª ""ÿÿÿÿÿÿÿÿÿ ¥
8__inference_batch_normalization_144_layer_call_fn_211735i,-./=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª ""ÿÿÿÿÿÿÿÿÿ î
S__inference_batch_normalization_145_layer_call_and_return_conditional_losses_211775?@ABM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 î
S__inference_batch_normalization_145_layer_call_and_return_conditional_losses_211793?@ABM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 É
S__inference_batch_normalization_145_layer_call_and_return_conditional_losses_211839r?@AB;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿII@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿII@
 É
S__inference_batch_normalization_145_layer_call_and_return_conditional_losses_211857r?@AB;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿII@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿII@
 Æ
8__inference_batch_normalization_145_layer_call_fn_211806?@ABM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Æ
8__inference_batch_normalization_145_layer_call_fn_211819?@ABM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¡
8__inference_batch_normalization_145_layer_call_fn_211870e?@AB;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿII@
p
ª " ÿÿÿÿÿÿÿÿÿII@¡
8__inference_batch_normalization_145_layer_call_fn_211883e?@AB;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿII@
p 
ª " ÿÿÿÿÿÿÿÿÿII@Ë
S__inference_batch_normalization_146_layer_call_and_return_conditional_losses_211923tRSTU<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ##
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ##
 Ë
S__inference_batch_normalization_146_layer_call_and_return_conditional_losses_211941tRSTU<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ##
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ##
 ð
S__inference_batch_normalization_146_layer_call_and_return_conditional_losses_211987RSTUN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ð
S__inference_batch_normalization_146_layer_call_and_return_conditional_losses_212005RSTUN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 £
8__inference_batch_normalization_146_layer_call_fn_211954gRSTU<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ##
p
ª "!ÿÿÿÿÿÿÿÿÿ##£
8__inference_batch_normalization_146_layer_call_fn_211967gRSTU<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ##
p 
ª "!ÿÿÿÿÿÿÿÿÿ##È
8__inference_batch_normalization_146_layer_call_fn_212018RSTUN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
8__inference_batch_normalization_146_layer_call_fn_212031RSTUN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿð
S__inference_batch_normalization_147_layer_call_and_return_conditional_losses_212071efghN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ð
S__inference_batch_normalization_147_layer_call_and_return_conditional_losses_212089efghN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ë
S__inference_batch_normalization_147_layer_call_and_return_conditional_losses_212135tefgh<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ë
S__inference_batch_normalization_147_layer_call_and_return_conditional_losses_212153tefgh<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 È
8__inference_batch_normalization_147_layer_call_fn_212102efghN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
8__inference_batch_normalization_147_layer_call_fn_212115efghN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ£
8__inference_batch_normalization_147_layer_call_fn_212166gefgh<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ£
8__inference_batch_normalization_147_layer_call_fn_212179gefgh<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿ»
G__inference_conv_layer1_layer_call_and_return_conditional_losses_211598p!"9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ¬¬
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ©© 
 
,__inference_conv_layer1_layer_call_fn_211607c!"9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ¬¬
ª ""ÿÿÿÿÿÿÿÿÿ©© »
G__inference_conv_layer2_layer_call_and_return_conditional_losses_211746p459¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_conv_layer2_layer_call_fn_211755c459¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ 
ª ""ÿÿÿÿÿÿÿÿÿ@¸
G__inference_conv_layer3_layer_call_and_return_conditional_losses_211894mGH7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿII@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿGG
 
,__inference_conv_layer3_layer_call_fn_211903`GH7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿII@
ª "!ÿÿÿÿÿÿÿÿÿGG¹
G__inference_conv_layer4_layer_call_and_return_conditional_losses_212042nZ[8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ##
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ!!
 
,__inference_conv_layer4_layer_call_fn_212051aZ[8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ##
ª "!ÿÿÿÿÿÿÿÿÿ!!à
W__inference_global_average_pooling2d_26_layer_call_and_return_conditional_losses_209978R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ·
<__inference_global_average_pooling2d_26_layer_call_fn_209984wR¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿð
M__inference_max_pooling2d_129_layer_call_and_return_conditional_losses_209513R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_max_pooling2d_129_layer_call_fn_209519R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿð
M__inference_max_pooling2d_130_layer_call_and_return_conditional_losses_209629R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_max_pooling2d_130_layer_call_fn_209635R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿð
M__inference_max_pooling2d_131_layer_call_and_return_conditional_losses_209745R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_max_pooling2d_131_layer_call_fn_209751R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿð
M__inference_max_pooling2d_132_layer_call_and_return_conditional_losses_209861R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_max_pooling2d_132_layer_call_fn_209867R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¨
H__inference_output_layer_layer_call_and_return_conditional_losses_212230\}~/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_output_layer_layer_call_fn_212239O}~/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "ÿÿÿÿÿÿÿÿÿá
I__inference_sequential_24_layer_call_and_return_conditional_losses_210548"!",-./45?@ABGHRSTUZ[efghqrwx}~F¢C
<¢9
/,
input_layerÿÿÿÿÿÿÿÿÿ¬¬
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 á
I__inference_sequential_24_layer_call_and_return_conditional_losses_210637"!",-./45?@ABGHRSTUZ[efghqrwx}~F¢C
<¢9
/,
input_layerÿÿÿÿÿÿÿÿÿ¬¬
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ü
I__inference_sequential_24_layer_call_and_return_conditional_losses_211184"!",-./45?@ABGHRSTUZ[efghqrwx}~A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ¬¬
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ü
I__inference_sequential_24_layer_call_and_return_conditional_losses_211313"!",-./45?@ABGHRSTUZ[efghqrwx}~A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ¬¬
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¹
.__inference_sequential_24_layer_call_fn_210800"!",-./45?@ABGHRSTUZ[efghqrwx}~F¢C
<¢9
/,
input_layerÿÿÿÿÿÿÿÿÿ¬¬
p

 
ª "ÿÿÿÿÿÿÿÿÿ¹
.__inference_sequential_24_layer_call_fn_210962"!",-./45?@ABGHRSTUZ[efghqrwx}~F¢C
<¢9
/,
input_layerÿÿÿÿÿÿÿÿÿ¬¬
p 

 
ª "ÿÿÿÿÿÿÿÿÿ´
.__inference_sequential_24_layer_call_fn_211386"!",-./45?@ABGHRSTUZ[efghqrwx}~A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ¬¬
p

 
ª "ÿÿÿÿÿÿÿÿÿ´
.__inference_sequential_24_layer_call_fn_211459"!",-./45?@ABGHRSTUZ[efghqrwx}~A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ¬¬
p 

 
ª "ÿÿÿÿÿÿÿÿÿÙ
$__inference_signature_wrapper_211045°"!",-./45?@ABGHRSTUZ[efghqrwx}~M¢J
¢ 
Cª@
>
input_layer/,
input_layerÿÿÿÿÿÿÿÿÿ¬¬";ª8
6
output_layer&#
output_layerÿÿÿÿÿÿÿÿÿ