# CODE EXAMPLE Ch.11 Asquith

require("lmomco")
require("beanplot")
# library("vioplot")

# load some data
dat <- read.csv('/Users/malindgren/Documents/repos/precip-dot/R_workspace/fbx_7day_ams_era-interim.csv', row.names=1)
FBX <- dat$'Fairbanks' # vectorize it

x <- list(FBX=FBX) # combine all into short variable name 

w <- sapply(x,length) # w will be used in a later example
# w <- length(FBX) # w will be used in a later example

# plots
boxplot(x, ylab="7-DAY ANNUAL MAX RAINFALL, IN INCHES", range=0)


# other fancy plots
rng <- sapply(x, range) # x from previous example
ylim <- c(min(rng[1,]), max(rng[2,]))
# par(mfrow=c(2,1), mai=c(0.5,1,0.5,0.5) )
beanplot(x, ll=0.04, main="BEAN PLOT: beanplot()", log="",
		ylim=ylim, ylab="7-DAY ANNUAL MAX RAINFALL,\n IN INCHES",
		overallline="median")

# get plotting positions
FBX.pp <- pp(FBX)
FBX.lmr <- lmoms(FBX)

# "afunc" <- function(r) {
#   return(c(FBX.lmr$ratios[r]))
# }

# L1 <- c(AMAR.lmr$lambdas[1], CANY.lmr$lambdas[1],
# 		CLAU.lmr$lambdas[1], HERF.lmr$lambdas[1], 
# 		TULA.lmr$lambdas[1], TUL6.lmr$lambdas[1], 
# 		VEGA.lmr$lambdas[1])*1.018; # bias correction factor Weiss(1964)

# T2 <- afunc(2); T3 <- afunc(3); T4 <- afunc(4)

# reg.L1  <- weighted.mean(L1, w);  reg.T2  <- weighted.mean(T2, w)
# reg.T3  <- weighted.mean(T3, w);  reg.T4  <- weighted.mean(T4, w)
# reg.lmr <- vec2lmom(c(reg.L1, reg.L1*reg.T2, reg.T3, reg.T4))
reg.kap <- parkap(FBX.lmr) # parameters of the Kappa distribution


# Ratio Diagram
lmrdia <- lmrdia()
plotlmrdia(lmrdia, autolegend=TRUE, nopoints=TRUE,
           nolimits=TRUE, xlim=c(0,0.3), ylim=c(-0.1,0.4),
           xleg=0.05, yleg=0.3)
# points(T3,T4)
# points(reg.T3,reg.T4, pch=16, cex=2)

# distributional analysis
F <- seq(0.001,0.999, by=0.001)
plot(F,quakap(F,reg.kap), type="n",
       # xlim=c(0,1), ylim=c(0,12),
       xlab="NONEXCEEDANCE PROBABILITY",
       ylab="7-DAY RAINFALL DEPTH, IN INCHES")

points(FBX.pp, FBX, pch=16, col=rgb(0, 0, 0, 0.15))
T.YEAR <- 100
PT <- quakap(T2prob(T.YEAR),reg.kap)
lines(c(T2prob(T.YEAR), T2prob(T.YEAR)), c(0,20), lty=2)
lines(c(0,1),c(PT,PT), lty=2)
lines(F,quakap(F,reg.kap), lwd=3)

