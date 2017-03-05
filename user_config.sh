#!/usr/bin/env bash
#
# This is a command line utility to configure the nrutils python package.
#
# Lionel London 2016 <lionel.london@ligo.org>
#

#
echo ">> This is a command line utility to configure the nrutils python package. All your base are belong to the Borg."

#
sed -i "s,%${var},${val},g" pipeline_gwfinj.ini
