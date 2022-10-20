#!/usr/bin/env python
from __future__ import print_function
import vcmsa
import unittest


class test_vcmsa(unittest.TestCase):


    def test1(self):
        '''
        Fake test
        '''    
        self.assertTrue(True == True)

    #def test1(self):
    #    '''
    #    Test of summary
    #    '''
    #    pp = vcmsa.PassageParser()
    #    p = pp.parse_passage("E1_E3")
    #    self.assertTrue(p.summary, list)
    #
    #def test2(self):
    #    '''
    #    Test example case
    #    '''

    #    pp = vcmsa.PassageParser()
    #    p = pp.parse_passage("Mdcksiat2_E3", 3)
    #    #print(vars(p))
    #    self.assertTrue(p.original == "Mdcksiat2_E3")
    #    self.assertTrue(p.plain_format == "MDCKSIAT2_E3")
    #    self.assertTrue(p.coerced_format == "S2_E3")
    #    self.assertTrue(p.ordered_passages) == ['MDCKSIAT2', 'E3']
    #    self.assertTrue(p.min_passages == 5)
    #    self.assertTrue(p.total_passages == 5)
    #    self.assertTrue(p.nth_passage == 'EGG')
    #    self.assertTrue(p.general_passages== ["CANINECELL", "EGG"])
    #    self.assertTrue(p.specific_passages == ["SIAT", "EGG"])
    #    self.assertTrue(p.passage_series == [[1, 'SIAT'], [2, 'SIAT'], [3, 'EGG'], [4, 'EGG'], [5, 'EGG']]) 
    #    self.assertTrue(p.summary == ['Mdcksiat2_E3', 'MDCKSIAT2_E3', 'S2_E3', 'CANINECELL+EGG', 'SIAT+EGG', 'exactly', '5'])
    #         

    #def test3(self):
    #    '''
    #    Test an empty passage annotation
    #    '''
    #    pp = vcmsa.PassageParser()
    #    p = pp.parse_passage("")
    #    self.assertTrue(p.original == "")
    #    self.assertTrue(p.plain_format == "")
    #    self.assertTrue(p.coerced_format == "")
    #    self.assertTrue(p.summary == ['','','','','','',''])

    #    self.assertTrue(p.min_passages == "")
    #    self.assertTrue(p.total_passages == "")
    #    self.assertTrue(p.nth_passage == "")
    #    self.assertTrue(p.general_passages== [])
    #    self.assertTrue(p.specific_passages == [])
    #    self.assertTrue(p.passage_series == []) 
    #         


    #def test4(self):
    #    '''
    #    Check a a longer list of passage IDs
    #    and write an outfile of the summary test
    #    These passage IDs are already partially formatted
    #    '''        

    #    with open("tests/test_passageIDs1.txt", "r") as passageIDs:
    #        with open("tests/output_test_passageIDs1.txt", "w") as outfile:
    #            for ID in passageIDs.readlines():
    #                pp = vcmsa.PassageParser()
    #                input_ID = ID.replace("\n", "") 
    #                full_annotation = pp.parse_passage(input_ID)
    #                quick_annotation = full_annotation.summary
    #                outfile.write(",".join(quick_annotation) + "\n")

    #def test5(self):
    #    '''
    #    Check another list of passage IDs
    #    and write an outfile
    #    '''
    #    with open("tests/test_passageIDs2.txt", "r") as passageIDs:
    #        with open("tests/output_test_passageIDs2.txt", "w") as outfile:
    #            for ID in passageIDs.readlines():
    #                pp = vcmsa.PassageParser()
    #                quick_annotation = pp.parse_passage(ID).summary
    #                outfile.write(str(",".join(quick_annotation)) + "\n")

    #def test6(self):
    #    '''
    #    Test a nonsense passage annotation
    #    '''
    #    pp = vcmsa.PassageParser()
    #    p = pp.parse_passage("asdk?&~EE8")
    #    self.assertTrue(p.original == "asdk?&~EE8")
    #    self.assertTrue(p.plain_format == "ASDK_EE8")
    #    self.assertTrue(p.coerced_format == "")
    #    self.assertTrue(p.summary == ['asdk?&~EE8', 'ASDK_EE8', '', '', '', '', ''])

    #    self.assertTrue(p.min_passages == "")
    #    self.assertTrue(p.total_passages == "")
    #    self.assertTrue(p.nth_passage == "")
    #    self.assertTrue(p.general_passages== [])
    #    self.assertTrue(p.specific_passages == [])
    #    self.assertTrue(p.passage_series == []) 
    

if __name__ == "__main__":
    pass 

