#!/usr/bin/perl

=head1 NAME

random-sample - Collect a random sample of data from the pipeline that can be
copied elsewhere for analysis.

=head1 SYNOPSIS

    random-sample [-h|--help] [-n SAMPLES] [-d|--dir DATA_DIR]

Randomly sample a number of different dataset/duration/time-period combinations
and report all of the data files used for each step of the pipeline for those
samples. That way, the output of this script can easily be piped into I<xargs> with
I<tar> or I<mv> or whatever to copy a collection of data that you can use to
analyze and QA the pipeline.

    Options:
        -h|--help:      Print help information and exit
        -n:             Specify the number of dataset/duration/time-period
                        combinations to sample. Defaults to 10.
        -d|--dir:       Specify the path to the data directory. Defaults to
                        /workspace/Shared/Tech_Projects/DOT/project_data/.

=cut

use Getopt::Long qw(GetOptions Configure);
use Pod::Usage qw(pod2usage);

# CPAN modules
use Math::Cartesian::Product;
use Sort::Naturally;


# Parse Options
my $sample_count = 10;
my $workdir      = '/workspace/Shared/Tech_Projects/DOT/project_data/';
Configure qw'auto_help';
GetOptions(
    'n=i'   => \$sample_count,
    'dir=s' => \$workdir
);
chdir $workdir or die "unable to cd to data directory $workdir: $!";

# Collect datasets, durations and time-periods/decades and build
# cartesian product to sample from.
my @datasets = qw(GFDL-CM3 NCAR-CCSM4);
my @durations = qw(
    60m 2h 3h 6h 12h 24h
    3d  4d 7d 10d 20d 30d 45d 60d
);
# The durations have slightly different names in the NOAA atlas files,
# we map the different ones here. (It's just 0-padding, really)
my %durations2noaa = (
    '60m'   => '01h',
    '2h'    => '02h',
    '3h'    => '03h',
    '6h'    => '06h'
);
my @decades = qw(
    2020-2029 2030-2039 2040-2049
    2050-2059 2060-2069 2070-2079
    2080-2089 2090-2099
);
my @combinations = cartesian {1} \@datasets, \@durations, \@decades;

# Collect a random sample
my @sample_combos;
for (1..$sample_count) {
    push @sample_combos, splice @combinations, rand @combinations, 1;
}

# Each pipeline step is associated with a function that will take, as arguments,
# a particular dataset, durations and time-period (in that order), and return
# a list of filenames associated with those parameters at that step.
# It's okay if files end up in the output multiple times, since we'll be
# remove duplicates at a later step.
my %step_map = (
    # The durations series contains one file per duration and dataset.
    # (But the smaller durations are split up by year)
    durations   => sub {
        my @files;

        # For small durations, split up by year
        if (grep {$_[1] eq $_} qw(60m 2h 3h 6h)) {
            # Get the file for each year of the decade.
            my @years = map {substr($_[2],0,3).$_} (0..9); # split decade into years
            for (@years) {
                my $pattern = sprintf 'durations/pcpt_%s_sum_wrf_%s_*_%s.nc', $_[1], $_[0], $_;
                push @files, glob($pattern);
            }
        }
        # For other durations, just get the projected and historical data for the duration
        else {
            my $pattern = sprintf 'durations/pcpt_%s_sum_wrf_%s_*.nc', $_[1], $_[0];
            push @files, glob($pattern);
        }

        return @files;
    },

    # The annual maximum series are split by durations, datasets and decades.
    ams         => sub {
        my @files;

        # Get projected data file.
        my $file = sprintf 'annual_maximum_series/pcpt_%s_rcp85_sum_wrf_%s_%s_ams.nc', @_;
        push @files, $file if -f $file;

        # Get historical data file (there should just be one regardless of decade)
        my $pattern = sprintf 'annual_maximum_series/pcpt_%s_historical_sum_wrf_%s_*_ams.nc', $_[0], $_[1];
        push @files, glob($pattern);

        return @files;
    },

    # The intervals are arranged and named the same basic way as the ams.
    intervals   => sub {
        my @files;

        # Get projected data file.
        my $file = sprintf 'output_interval_durations/pcpt_%s_rcp85_sum_wrf_%s_%s_intervals.nc', @_;
        push @files, $file if -f $file;

        # Get historical data file (there should just be one regardless of decade)
        my $pattern = sprintf 'output_interval_durations/pcpt_%s_historical_sum_wrf_%s_*_intervals.nc', $_[0], $_[1];
        push @files, glob($pattern);

        return @files;
    },

    # The deltas are the first step to not have projected and historical data
    # (since they're combined)
    deltas   => sub {
        # Get data file.
        my $file = sprintf 'deltas/pcpt_%s_sum_wrf_%s_%s_deltas.nc', @_;
        return $file if -f $file;
        return ();
    },

    # Warped results are named the same way as the deltas
    warp   => sub {
        # Get data file.
        my $file = sprintf 'warped/pcpt_%s_sum_wrf_%s_%s_warped.nc', @_;
        return $file if -f $file;
        return ();
    },

    # Combined results are named the same way as the deltas
    multiply   => sub {
        # Get data file.
        my $file = sprintf 'combined/pcpt_%s_sum_wrf_%s_%s_combined.nc', @_;
        return $file if -f $file;
        return ();
    }
);

my @all_files;
for my $combo (@sample_combos) {
    for (keys %step_map) {
        my @files = $step_map{$_}->(@$combo);
        push @all_files, @files;
    }
}

# Remove duplicate files
my %file_hash = map {$_,1} @all_files;
@all_files = keys %file_hash;
# Sort
@all_files = nsort @all_files;

print "$_\n" for @all_files;
