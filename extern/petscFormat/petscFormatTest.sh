#!/bin/bash

#Clean up any old test results
rm -f checkbadSource.out

echo "Tabs in file -----------------------------------------" > checkbadSource.out
git --no-pager grep -l -P  '\t'  -- '*.[chF]' '*.F90' >> checkbadSource.out;true

echo "White space at end of line ---------------------------" >> checkbadSource.out
git --no-pager grep -l -P ' $$' -- '*.[chF]' '*.F90'  >> checkbadSource.out;true

echo "Space after ( ----------------------------------------" >> checkbadSource.out
git --no-pager grep -l -P '\( ' -- '*.[chF]' '*.F90' | grep PetscErrorCode  >> checkbadSource.out;true
git --no-pager grep -l -P 'if \( ' -- '*.[chF]' '*.F90'   >> checkbadSource.out;true
git --no-pager grep -l -P 'for \( ' -- '*.[chF]' '*.F90'  >> checkbadSource.out;true
git --no-pager grep -l -P 'while \( ' -- '*.[chF]' '*.F90'  >> checkbadSource.out;true

echo "Space before ) ---------------------------------------" >> checkbadSource.out
git --no-pager grep -l -P ' \)' -- '*.[chF]' '*.F90'  >> checkbadSource.out;true

echo "Space before CHKERRQ ---------------------------------" >> checkbadSource.out
git --no-pager grep -l -P '; CHKERRQ\(' -- '*.[chF]' '*.F90'  >> checkbadSource.out;true
git --no-pager grep -l -P ' +CHKERRQ\(' -- '*.[chF]' '*.F90'  >> checkbadSource.out;true

echo "No space after if, for or while -----------------------------" >> checkbadSource.out
git --no-pager grep -l -P -e '[^_]for\(' -e 'if\('  -e 'while\(' -- '*.[chF]' '*.F90' >> checkbadSource.out;true

echo "Two ;; -----------------------------" >> checkbadSource.out
git --no-pager grep  -P -e ';;'  -- '*.[chF]' '*.F90' | grep -v ' for (' >> checkbadSource.out;true

echo "Missing if (ierr) return(ierr); -----------------------------" >> checkbadSource.out
git --no-pager grep  -P -e 'ierr = PetscInitialize\('  -- '*.[ch]' | grep -v 'if (ierr) return ierr;' | egrep "(test|tutorial)" >> checkbadSource.out;true

echo "DOS file (with DOS newlines)--------------------------"  >> checkbadSource.out
git --no-pager grep  -l -P '\r' -- '*.[chF]' '*.F90'  >> checkbadSource.out;true

echo "{ before SETERRQ --------------------------" >> checkbadSource.out
git --no-pager grep  -l -P '{SETERRQ' -- '*.[chF]' '*.F90' >> checkbadSource.out;true

a=`cat checkbadSource.out | wc -l`; let "l=$a-10" ;
   if [ $l -gt 0 ] ; then
     echo $l " files with errors detected in source code formatting" ;
     cat checkbadSource.out ;
     exit 1
   fi;
git --no-pager grep  -P  -n "[\x00-\x08\x0E-\x1F\x80-\xFF]"  -- '*.[chF]' '*.F90' > badSourceChar.out;
   w=`cat badSourceChar.out | wc -l`;
   if [ $w -gt 0 ] ; then
     echo "Source files with non-ASCII characters ----------------" ;
     cat badSourceChar.out ;
     exit 1
   fi
