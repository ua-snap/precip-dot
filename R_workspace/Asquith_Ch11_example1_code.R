# CODE EXAMPLE Ch.11 Asquith

require("lmomco")
library("beanplot")
library("vioplot")

# load some data
data(amarilloprecip) # from lmomco package
data(canyonprecip)
data(claudeprecip) 
data(herefordprecip)
data(tuliaprecip)
data(tulia6Eprecip)
data(vegaprecip)

AMAR <- sort(amarilloprecip$DEPTH)
CANY <- sort(canyonprecip$DEPTH)
CLAU <- sort(claudeprecip$DEPTH)
HERF <- sort(herefordprecip$DEPTH)
TULA <- sort(tuliaprecip$DEPTH)
TUL6 <- sort(tulia6Eprecip$DEPTH)
VEGA <- sort(vegaprecip$DEPTH)

# /Users/malindgren/Documents/repos/precip-dot/R_workspace/fbx_7day_ams_era-interim.csv

x <- list(AMAR=AMAR, CANY=CANY, CLAU=CLAU,
          HERF=HERF, TULA=TULA, TUL6=TUL6,
			VEGA=VEGA) # combine all into short variable name 
w <- sapply(x,length) # w will be used in a later example

# plots
boxplot(x, ylab="7-DAY ANNUAL MAX RAINFALL, IN INCHES", range=0)


# other fancy plots
rng <- sapply(x, range) # x from previous example
ylim <- c(min(rng[1,]), max(rng[2,]))
par(mfrow=c(2,1), mai=c(0.5,1,0.5,0.5) )
beanplot(x, ll=0.04, main="BEAN PLOT: beanplot()", log="",
		ylim=ylim, ylab="7-DAY ANNUAL MAX RAINFALL,\n IN INCHES",
		overallline="median")
cities <- names(x); data <- x # get names and make a copy 
names(data)[1] <- "x" # modify the copy 
do.call("vioplot",c(data, list(ylim=ylim, names=cities, col="white")))

title(main="VIOLIN PLOT: vioplot()",
      ylab="7-DAY ANNUAL MAX RAINFALL,\n IN INCHES")


# get plotting positions
AMAR.pp <- pp(AMAR);  CANY.pp <- pp(CANY)
CLAU.pp <- pp(CLAU);  HERF.pp <- pp(HERF)
TULA.pp <- pp(TULA);  TUL6.pp <- pp(TUL6)
VEGA.pp <- pp(VEGA)
AMAR.lmr <- lmoms(AMAR);  CANY.lmr <- lmoms(CANY)
CLAU.lmr <- lmoms(CLAU);  HERF.lmr <- lmoms(HERF)
TULA.lmr <- lmoms(TULA);  TUL6.lmr <- lmoms(TUL6)
VEGA.lmr <- lmoms(VEGA)

"afunc" <- function(r) {
  return(c(AMAR.lmr$ratios[r], CANY.lmr$ratios[r],
           CLAU.lmr$ratios[r], HERF.lmr$ratios[r],
           TULA.lmr$ratios[r], TUL6.lmr$ratios[r],
           VEGA.lmr$ratios[r]))
}

L1 <- c(AMAR.lmr$lambdas[1], CANY.lmr$lambdas[1],
		CLAU.lmr$lambdas[1], HERF.lmr$lambdas[1], 
		TULA.lmr$lambdas[1], TUL6.lmr$lambdas[1], 
		VEGA.lmr$lambdas[1])*1.018; # bias correction factor Weiss(1964)

T2 <- afunc(2); T3 <- afunc(3); T4 <- afunc(4)


reg.L1  <- weighted.mean(L1, w);  reg.T2  <- weighted.mean(T2, w)
reg.T3  <- weighted.mean(T3, w);  reg.T4  <- weighted.mean(T4, w)
reg.lmr <- vec2lmom(c(reg.L1, reg.L1*reg.T2, reg.T3, reg.T4))
reg.kap <- parkap(reg.lmr) # parameters of the Kappa distribution


# Ratio Diagram
lmrdia <- lmrdia()
plotlmrdia(lmrdia,autolegend=TRUE, nopoints=TRUE,
           nolimits=TRUE, xlim=c(0,0.3), ylim=c(-0.1,0.4),
           xleg=0.05, yleg=0.3)
points(T3,T4)
points(reg.T3,reg.T4, pch=16, cex=2)

# distributional analysis
F <- seq(0.001,0.999, by=0.001)
plot(F,quakap(F,reg.kap), type="n",
       xlim=c(0,1), ylim=c(0,12),
       xlab="NONEXCEEDANCE PROBABILITY",
       ylab="7-DAY RAINFALL DEPTH, IN INCHES")
points(AMAR.pp, AMAR, pch=16, col=rgb(0, 0, 0, 0.15))
points(CANY.pp, CANY, pch=16, col=rgb(0, 0, 0, 0.20))
points(CLAU.pp, CLAU, pch=16, col=rgb(0, 0, 0, 0.25))
points(HERF.pp, HERF, pch=16, col=rgb(0, 0, 0, 0.35))
points(TULA.pp, TULA, pch=16, col=rgb(0, 0, 0, 0.40))
points(TUL6.pp, TUL6, pch=16, col=rgb(0, 0, 0, 0.45))
points(VEGA.pp, VEGA, pch=16, col=rgb(0, 0, 0, 0.50))
T.YEAR <- 100
PT <- quakap(T2prob(T.YEAR),reg.kap)
lines(c(T2prob(T.YEAR), T2prob(T.YEAR)), c(0,20), lty=2)
lines(c(0,1),c(PT,PT), lty=2)
lines(F,quakap(F,reg.kap), lwd=3)

