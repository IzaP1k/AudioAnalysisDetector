import pytest


class Class:
    def __init__(self, subject, day, start_time, end_time, building, room, professor):
        self.subject = subject
        self.day = day
        self.start_time = start_time
        self.end_time = end_time
        self.building = building
        self.room = room
        self.professor = professor

class Student:
    def __init__(self, name, surname, index):
        self.name = name
        self.surname = surname
        self.index = index
        self.classes = []

    def enroll_in_class(self, university_class):
        self.classes.append([university_class, 0.0])

    def leave_class(self, subject_name, start_time, professor_name):
        for enrolled_class, grade in self.classes:
            if (
                enrolled_class.subject == subject_name
                and enrolled_class.start_time == start_time
                and enrolled_class.professor == professor_name
            ):
                self.classes.remove([enrolled_class, grade])
                return

    def check_class_collisions(self):
        for i in range(len(self.classes)):
            for j in range(i + 1, len(self.classes)):
                if (
                    self.classes[i][0].day == self.classes[j][0].day
                    and self.classes[i][0].end_time < self.classes[j][0].end_time
                    and self.classes[i][0].end_time > self.classes[j][0].start_time
                ):
                    return True
        return False

    def grade_class(self, class_name, professor_name, grade):
        for enrolled_class, student_grade in self.classes:
            if (enrolled_class.subject == class_name):
                student_grade = grade

    def get_average_grade(self):
        total_grade = 0
        total_subjects = 0
        for university_class, grade in self.classes:
            total_grade += grade
            total_subjects += 1
        if total_subjects == 0:
            return 0
        return round(total_grade / total_subjects, 4)

    def list_professors(self):
        professors = []
        for university_class, grade in self.classes:
            professors.append(university_class.professor)
        return list(set(professors))

    def list_shared_groups(self, student2):
        shared_groups = []
        for class1, grade1 in self.classes:
            for class2, grade2 in student2.classes:
                if class1 == class2:
                    shared_groups.append(class1)
        return shared_groups

class University:
    def __init__(self, name):
        self.name = name
        self.classes = []
        self.students = []

    def add_student(self, student):
        for existing_student in self.students:
            if existing_student.index == student.index:
                return False
        self.students.append(student)
        return True

    def delete_student(self, index):
        for student in self.students:
            if student.index == index:
                self.students.remove(student)
                return True
        return False

    def add_class(self, university_class):
        for existing_class in self.classes:
            if (
                existing_class.subject == university_class.subject
                and existing_class.day == university_class.day
                and existing_class.start_time == university_class.start_time
                and existing_class.building == university_class.building
                and existing_class.room == university_class.room
            ):
                return False
        self.classes.append(university_class)
        return True

    def delete_class(self, subject_name, start_time, professor_name):
        for university_class in self.classes:
            if (
                university_class.subject == subject_name
                and university_class.start_time == start_time
                and university_class.professor == professor_name
            ):
                self.classes.remove(university_class)
                return True
        return False

    def timetable(self, day):
        day_classes = []
        for university_class in self.classes:
            if university_class.day == day:
                day_classes.append(
                    (
                        university_class.start_time,
                        university_class.end_time,
                        university_class.subject,
                    )
                )
        day_classes.sort(key=lambda x: x[0])
        return day_classes

    def student_ranking(self):
        ranking = []
        for student in self.students:
            average_grade = student.get_average_grade()
            ranking.append(
                (
                    f"{student.name} {student.surname}",
                    student.index,
                    round(average_grade, 3),
                )
            )
        ranking.sort(key=lambda x: x[2], reverse=True)
        return ranking

    def get_active_buildings(self):
        buildings = []
        for university_class in self.classes:
            buildings.append(university_class.building)
        return buildings

# 2.2.1 - enroll_in_class
def test_enroll_studentNoAttend():
    student1 = Student("Jan", "Kowalski", 123456)
    class1 = Class("User experience", "monday", (7, 30), (9, 00), "A1", "127", "Adam Nowak")
    class2 = Class("Math", "monday", (7, 30), (9, 00), "A1", "127", "Adam Nowak")

    student1.enroll_in_class(class1)

    all_classes = []
    if len(student1.classes) != 0:
        for class_mark in student1.classes:
            all_classes.append(class_mark[0])

    assert class2 not in all_classes


def test_enroll_objectClass():
    student1 = Student("Jan", "Kowalski", 123456)
    class1 = Class("User experience", "monday", (7, 30), (9, 00), "A1", "127", "Adam Nowak")
    student1.enroll_in_class(class1)
    assert student1.classes[-1][0] is class1


def test_enroll_studentAttend():
    student1 = Student("Jan", "Kowalski", 123456)
    class1 = Class("User experience", "monday", (7, 30), (9, 00), "A1", "127", "Adam Nowak")

    student1.enroll_in_class(class1)

    all_classes = []
    if len(student1.classes) != 0:
        for class_mark in student1.classes:
            all_classes.append(class_mark[0])

    assert class1 in all_classes

#
# # # 2.2.2 - leave_class
#
# def test_leave_class_studentAttend():
#     student1 = Student("Jan", "Kowalski", 123456)
#     class1 = Class("User experience", "monday", (7, 30), (9, 00), "A1", "127", "Adam Nowak")
#     student1.enroll_in_class(class1)
#     student1.leave_class(class1.subject, class1.start_time, class1.professor)
#
#     student1.leave_class(class1.subject, class1.start_time, class1.professor)
#
#     all_classes = []
#     if len(student1.classes) != 0:
#         for class_mark in student1.classes:
#             all_classes.append(class_mark[0])
#
#     assert class1 not in all_classes
#
#
# def test_leave_class_studentNotAttend():
#     student1 = Student("Jan", "Kowalski", 123456)
#     class1 = Class("User experience", "monday", (7, 30), (9, 00), "A1", "127", "Adam Nowak")
#     class2 = Class("Math", "tuesday", (7, 40), (9, 00), "A2", "127", "Adam Nowak")
#     student1.enroll_in_class(class1)
#
#     classes_before = student1.classes
#
#     student1.leave_class(class2.subject, class2.start_time, class2.professor)
#
#     classes_after = student1.classes
#
#     assert classes_before == classes_after
# #
# #
# # # 2.2.3
#
# def test_check_class_collisions_False():
#     student1 = Student("Jan", "Kowalski", 123456)
#     class1 = Class("User experience", "monday", (7, 30), (9, 00), "A1", "127", "Adam Nowak")
#     class2 = Class("Math", "tuesday", (7, 40), (9, 00), "A2", "127", "Adam Nowak")
#     student1.enroll_in_class(class1)
#     student1.enroll_in_class(class2)
#
#     assert student1.check_class_collisions() == False
#
#
# def test_check_class_collisions_True():
#     student1 = Student("Jan", "Kowalski", 123456)
#     class1 = Class("User experience", "monday", (7, 30), (9, 00), "A1", "127", "Adam Nowak")
#     class2 = Class("Math", "monday", (7, 40), (9, 00), "A2", "127", "Adam Nowak")
#     student1.enroll_in_class(class1)
#     student1.enroll_in_class(class2)
#
#
#     assert student1.check_class_collisions() == True
#
#
# # 2.2.4
#
# def test_grade_class_NewCorrect():
#     grade = 2.0
#     class_name = "User experience"
#
#     student1 = Student("Jan", "Kowalski", 123456)
#     class1 = Class(class_name, "monday", (7, 30), (9, 00), "A1", "127", "Adam Nowak")
#     student1.enroll_in_class(class1)
#     student1.grade_class(class1.subject, class1.professor, grade)
#
#     all_classes = []
#     all_marks = []
#     if len(student1.classes) != 0:
#         for class_mark in student1.classes:
#             all_classes.append(class_mark[0])
#             all_marks.append(class_mark[1])
#
#
#     if class1 in all_classes:
#         idx = all_classes.index(class1)
#         assert all_marks[idx] == grade
#     else:
#         assert False
#
#
#
# def test_grade_class_NewIncorrect():
#     grade = 10.0
#
#     student1 = Student("Jan", "Kowalski", 123456)
#     class1 = Class("User experience", "monday", (7, 30), (9, 00), "A1", "127", "Adam Nowak")
#     student1.enroll_in_class(class1)
#
#     all_classes = []
#     all_marks = []
#     if len(student1.classes) != 0:
#         for class_mark in student1.classes:
#             all_classes.append(class_mark[0])
#             all_marks.append(class_mark[1])
#
#
#     if class1 in all_classes:
#         idx = all_classes.index(class1)
#         assert all_marks[idx] != grade
#     else:
#         assert False
#
#
# def test_grade_class_AlreadyMark():
#     grade = 2.0
#     grade2 = 3.0
#
#     student1 = Student("Jan", "Kowalski", 123456)
#     class1 = Class("User experience", "monday", (7, 30), (9, 00), "A1", "127", "Adam Nowak")
#     student1.enroll_in_class(class1)
#     student1.grade_class(class1.subject, class1.professor, grade)
#
#     student1.grade_class(class1.subject, class1.professor, grade2)
#
#     all_classes = []
#     all_marks = []
#     if len(student1.classes) != 0:
#         for class_mark in student1.classes:
#             all_classes.append(class_mark[0])
#             all_marks.append(class_mark[1])
#
#     if class1 in all_classes:
#         idx = all_classes.index(class1)
#         assert all_marks[idx] != grade2
#     else:
#         assert False
#
#
# # 2.2.5.
#
# def test_get_average_grade_correctMarks():
#     grade = 2.0
#     grade2 = 3.0
#     grade3 = 5.0
#
#     student1 = Student("Jan", "Kowalski", 123456)
#     class1 = Class("User experience", "monday", (7, 30), (9, 00), "A1", "127", "Adam Nowak")
#     class2 = Class("Math", "monday", (7, 40), (9, 00), "A2", "127", "Adam Nowak")
#     class3 = Class("Art", "friday", (7, 40), (9, 00), "A5", "127", "Adam Drapała")
#
#     student1.classes = [[class1, grade], [class2, grade2], [class3, grade3]]
#
#     assert student1.get_average_grade() == 3.3333
#
#
# def test_get_average_grade_withzero():
#     grade = 2.0
#     grade2 = 3.0
#     grade3 = 0.0
#
#     student1 = Student("Jan", "Kowalski", 123456)
#     class1 = Class("User experience", "monday", (7, 30), (9, 00), "A1", "127", "Adam Nowak")
#     class2 = Class("Math", "monday", (7, 40), (9, 00), "A2", "127", "Adam Nowak")
#     class3 = Class("Art", "friday", (7, 40), (9, 00), "A5", "127", "Adam Drapała")
#
#     student1.classes = [[class1, grade], [class2, grade2], [class3, grade3]]
#
#     assert student1.get_average_grade() == 2.5
#
# #2.2.6
#
# def test_list_professors_oneProfessor():
#
#     student1 = Student("Jan", "Kowalski", 123456)
#     class3 = Class("Art", "friday", (7, 40), (9, 00), "A5", "127", "Adam Drapała")
#
#     student1.enroll_in_class(class3)
#
#     assert student1.list_professors() == ["Adam Drapała"]
#
#
# def test_list_professors_twoProfessors():
#     student1 = Student("Jan", "Kowalski", 123456)
#
#     class3 = Class("Art", "friday", (7, 40), (9, 00), "A5", "127", "Adam Drapała")
#     class2 = Class("Math", "monday", (7, 40), (9, 00), "A2", "127", "Adam Nowak")
#
#     student1.enroll_in_class(class3)
#     student1.enroll_in_class(class2)
#
#     assert student1.list_professors() == ["Adam Drapała", "Adam Nowak"]
#
#
# def test_list_professors_oneProfessorTwice():
#     student1 = Student("Jan", "Kowalski", 123456)
#
#     class3 = Class("Art", "friday", (7, 40), (9, 00), "A5", "127", "Adam Nowak")
#     class2 = Class("Math", "monday", (7, 40), (9, 00), "A2", "127", "Adam Nowak")
#
#     student1.enroll_in_class(class3)
#     student1.enroll_in_class(class2)
#
#     assert student1.list_professors() == ["Adam Nowak"]
#
# def test_list_professors_zeroProfessor():
#
#     student1 = Student("Jan", "Kowalski", 123456)
#
#     assert student1.list_professors() == []
#
# #2.2.7
#
# def test_list_shared_groups_NoTogether():
#     student1 = Student("Jan", "Kowalski", 123456)
#     student2 = Student("Kuba", "Nowak", 456098)
#
#     class3 = Class("Art", "friday", (7, 40), (9, 00), "A5", "127", "Adam Drapała")
#     class2 = Class("Math", "monday", (7, 40), (9, 00), "A2", "127", "Adam Nowak")
#
#     student1.enroll_in_class(class3)
#     student2.enroll_in_class(class2)
#
#     assert student1.list_shared_groups(student2) == []
#
# def test_list_shared_groups_Together():
#     student1 = Student("Jan", "Kowalski", 123456)
#     student2 = Student("Kuba", "Nowak", 456098)
#
#     class3 = Class("Art", "friday", (7, 40), (9, 00), "A5", "127", "Adam Drapała")
#     class2 = Class("Math", "monday", (7, 40), (9, 00), "A2", "127", "Adam Nowak")
#
#     student1.enroll_in_class(class3)
#     student1.enroll_in_class(class2)
#     student2.enroll_in_class(class2)
#
#     assert student1.list_shared_groups(student2) == [class2]
#
# def test_list_shared_groups_TogetherTwoGroups():
#     student1 = Student("Jan", "Kowalski", 123456)
#     student2 = Student("Kuba", "Nowak", 456098)
#
#     class3 = Class("Art", "friday", (7, 40), (9, 00), "A5", "127", "Adam Drapała")
#     class2 = Class("Math", "monday", (7, 40), (9, 00), "A2", "127", "Adam Nowak")
#
#     student1.enroll_in_class(class3)
#     student1.enroll_in_class(class2)
#     student2.enroll_in_class(class2)
#     student2.enroll_in_class(class3)
#
#     assert student1.list_shared_groups(student2) == [class2, class3]

# UNIVERSITY




# TWO_STUDENTS_INDEX = 1234
#
#
# @pytest.fixture
# def university_two_students_with_the_same_index():
# university = University("PWr")
# student_1 = Student("Jan", "Kowalski", TWO_STUDENTS_INDEX)
# university.add_student(student_1)
# return university
#
#
# @pytest.fixture
# def dummy_student():
# return Student("Piotr", "Nowak", TWO_STUDENTS_INDEX)
#
#
# @pytest.fixture
# def student2():
# return Student("Anna", "Kwiat", 343434)
#
#
# @pytest.fixture
# def student3():
# return Student("Ola", "Lilia", 45654)
#
#
#
#
# # 3.2.1
def test_add_student_with_the_same_index(university_two_students_with_the_same_index, dummy_student):
    test_university = university_two_students_with_the_same_index
    assert False == test_university.add_student(dummy_student)


def test_add_student_with_different_index(university_two_students_with_the_same_index):
    test_university = university_two_students_with_the_same_index
    student_2 = Student("Marta", "Kowalska", 7654)
    assert True == test_university.add_student(student_2)
#
#
# # 3.2.2
# def test_delete_student_success(university_two_students_with_the_same_index):
# dummy_university = university_two_students_with_the_same_index
# assert True == dummy_university.delete_student(TWO_STUDENTS_INDEX)
#
#
# def test_delete_student_not_exists(university_two_students_with_the_same_index):
# dummy_university = university_two_students_with_the_same_index
# assert False == dummy_university.delete_student("999")
#
#
# # 3.2.3
# @pytest.fixture
# def class1():
# return Class("ENGLISH", "Monday", (10,30), (11,30), "B4", "4.44", "Piast")
#
#
# @pytest.fixture
# def class2():
# return Class("POLISH", "Monday", (10,30), (11,30), "B4", "4.44", "Piast")
#
#
# @pytest.fixture
# def class3():
# return Class("Spanish", "Friday", (9,15), (10,00), "A1", "5.2", "Kot")
#
#
# @pytest.fixture
# def class4():
# return Class("WF", "Monday", (12,15), (13,00), "A1", "1.11", "Pies")
#
#
# def test_add_class_success(university_two_students_with_the_same_index, class1):
# dummy_university = university_two_students_with_the_same_index
# assert True == dummy_university.add_class(class1)
#
#
# def test_add_class_overlapping(university_two_students_with_the_same_index, class1, class2):
# dummy_university = university_two_students_with_the_same_index
# dummy_university.add_class(class1)
# assert False == dummy_university.add_class(class2)
#
#
# #3.2.4.
#
#
# def test_delete_class_success(university_two_students_with_the_same_index, class1, class2):
# dummy_university = university_two_students_with_the_same_index
# dummy_university.classes = [class1]
# assert True == dummy_university.delete_class("ENGLISH", (10,30), "Piast")
#
#
# def test_delete_class_lack_of_classes(university_two_students_with_the_same_index, class1):
# dummy_university = university_two_students_with_the_same_index
# assert False == dummy_university.delete_class("English", (10,30), "Piast")
#
#
# #3.2.5.
#
#
# def test_timetable(university_two_students_with_the_same_index, class1, class3, class4):
# dummy_university = university_two_students_with_the_same_index
# dummy_university.classes = [class1, class3, class4]
# expected_output = [((10,30), (11,30), "ENGLISH"), ((12,15), (13,00),"WF")]
# assert expected_output == dummy_university.timetable("Monday")
#
#
# def test_timetable_freeday(university_two_students_with_the_same_index):
# dummy_university = university_two_students_with_the_same_index
# expected_output = []
# assert dummy_university.timetable("Thursday") == expected_output
#
#
# #3.2.6.
#
#
# def test_student_ranking_less_than_3(university_two_students_with_the_same_index, dummy_student, student2, class1, class4):
# dummy_university = university_two_students_with_the_same_index
# student2.classes = [[class1, 5.0], [class4, 4.0]]
# dummy_student.classes = [[class1, 3.0], [class4, 3.5]]
# dummy_university.students = [dummy_student, student2]
# assert dummy_university.student_ranking() == []
#
#
#
#
# def test_student_ranking_success(university_two_students_with_the_same_index, dummy_student, student2, student3, class1, class4):
# dummy_university = university_two_students_with_the_same_index
# student2.classes = [[class1, 5.0], [class4, 4.0]]
# student3.classes = [[class1, 3.0], [class4, 3.0]]
# dummy_student.classes = [[class1, 3.0], [class4, 3.5]]
# dummy_university.students = [dummy_student, student2, student3]
# expected_output = [("Anna Kwiat", 343434, 4.500,), ("Piotr Nowak", 1234, 3.250), ("Ola Lilia", 45654, 3.000)]
# assert expected_output == dummy_university.student_ranking()
#
#
#
#
# # 3.2.7
#
#
# def test_get_active_buildings_NoStudent():
#
#
# university = University("PWr")
#
#
# class3 = Class("Art", "friday", (7, 40), (9, 00), "A5", "127", "Adam Drapała")
# class2 = Class("Math", "monday", (7, 40), (9, 00), "A2", "127", "Adam Nowak")
#
#
# university.add_class(class3)
# university.add_class(class2)
#
#
# assert university.get_active_buildings() == []
#
#
# def test_get_active_buildings_OneBuilding():
#
#
# university = University("PWr")
#
#
# class3 = Class("Art", "friday", (7, 40), (9, 00), "A2", "127", "Adam Drapała")
# class2 = Class("Math", "monday", (7, 40), (9, 00), "A2", "127", "Adam Nowak")
#
#
# university.add_class(class3)
# university.add_class(class2)
#
#
# student1 = Student("Jan", "Kowalski", 123456)
# student2 = Student("Kuba", "Nowak", 456098)
#
#
# student1.enroll_in_class(class3)
# student1.enroll_in_class(class2)
# student2.enroll_in_class(class2)
#
#
# assert university.get_active_buildings() == ["A2"]
#
#
# def test_get_active_buildings_TwoBuilding():
#
#
# university = University("PWr")
#
#
# class3 = Class("Art", "friday", (7, 40), (9, 00), "A2", "127", "Adam Drapała")
# class2 = Class("Math", "monday", (7, 40), (9, 00), "A4", "127", "Adam Nowak")
#
#
# university.add_class(class3)
# university.add_class(class2)
#
#
# student1 = Student("Jan", "Kowalski", 123456)
# student2 = Student("Kuba", "Nowak", 456098)
#
#
# student1.enroll_in_class(class3)
# student1.enroll_in_class(class2)
# student2.enroll_in_class(class2)
#
#
# assert university.get_active_buildings() == ["A2", "A4"]