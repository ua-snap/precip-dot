#!/usr/bin/perl
use strict;
use warnings;

=head1 NAME

run-pipeline - Execute all or part of the data-processing pipeline

=head1 SYNOPSIS

run-pipeline [-h|--help] [-n|--dry-run] DATA_DIR LIST
run-pipeline DATA_DIR ls

Options:
    -h|--help:      Print this help and exit
    -n|--dry-run:   Show which steps of the pipeline will get run
                    but don't actually run them.

DATA_DIR is the path to the directory where the data for the pipeline lies.

LIST is a comma-separated list of steps to run in the pipeline. The format
is analgous to the field selection of the cut(1) command. Each element of the
list can be either the name of a step, or a range in one of the formats
below:
    NAME        The single step with NAME
    NAME-       The step with NAME and every step after it
    NAME-NAME2  All the steps from NAME to NAME2 (inclusive)
    -NAME       The steps from the first one up through NAME
Every step will only be run once and always in the order that the script
defines for the pipeline, REGARDLESS of how you specify them in the list.

Execute `run-pipeline ls` to see a list of the names for all the steps in
the pipeline and a description of each step. In reality 'ls' is just another 
step in the pipeline that does nothing but print the names of all the steps.

=cut

# Mapping of steps to descriptions
my %descriptions = (
    'ls'        => "List all steps in the pipeline.",
    'durations' => "Compute durations series from raw hourly data",
    'ams'       => "Compute annual maximum series form durations series",
    'intervals' => "Compute return intervals (with confidence bounds) from annual maximum series"
);

# Ordering of steps
my @order = qw(ls durations ams intervals);

# Subroutines for each pipeline step

sub list_steps {
    print "The steps of the pipeline are (in order):\n";
    foreach (0..$#order) {
        printf "\033[1m%d: %-10s\033[0m\t%s\n", $_+1, $order[$_], $descriptions{$order[$_]};
    }
}

# Mapping of steps to subroutines
my %subs = (
    'ls'        => \&list_steps,
    'durations' => undef,
    'ams'       => undef,
    'intervals' => undef
);

#######################
# MAIN
#######################

use Getopt::Long qw(GetOptions Configure);
use Pod::Usage qw(pod2usage);

# Parse Options
my $dry_run = 0;
Configure qw'auto_help pass_through';
GetOptions(
    'dry-run|n' => \$dry_run
);
pod2usage("Invalid arguments.") if (@ARGV != 2);

(my $data_dir, my $step_list) = @ARGV;

die "Data directory, $data_dir, is not a directory." unless (-d $data_dir);

# Parse steps
use List::Util qw(first);

# Get the order index of the specified step
# (or die if the name doesn't match anything)
sub get_step {
    my $step = shift @_;
    my $idx = first { $order[$_] eq $step } 0..$#order;
    unless (defined($idx)) {
        print STDERR "Invalid step name! '$step'\n";
        list_steps;
        exit 2;
    }
    return $idx;
}

my @do_step = (0)x@order; # Flags indicating whether each step should be performed
foreach (split /,/, $step_list) {
    my @slice;
    if (/^\w+$/) {
        @slice = get_step($_);
    }
    elsif (/^-(\w+)$/) {
        @slice = 0..get_step($1);
    }
    elsif (/^(\w+)-$/) {
        @slice = get_step($1)..$#do_step;
    }
    elsif (/^(\w+)-(\w+)$/) {
        @slice = get_step($1)..get_step($2);
    }
    else {
        pod2usage("Invalid LIST element!");
    }
    @do_step[@slice] = (1)x@slice;
}

# Execute steps

if ($dry_run) {
    foreach (0..$#do_step) {
        print "${order[$_]}\n" if $do_step[$_];
    }
    exit 0;
}

foreach (0..$#do_step) {
    $subs{$order[$_]}->() if $do_step[$_];
}