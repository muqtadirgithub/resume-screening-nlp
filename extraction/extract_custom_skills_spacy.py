
import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab)

skills_list = [
    # Legal & Law
    "Legal Research", "Client Representation", "Contract Drafting", "Courtroom Advocacy", "Negotiation",
    "Legal Writing", "Litigation", "Legal Compliance", "Case Management", "Civil Law",
    "Criminal Law", "Intellectual Property", "Legal Ethics", "Corporate Law", "Family Law",
    "Legal Discovery", "Legal Technology", "Mediation", "Regulatory Compliance", "Paralegal Support",

    # Fine Arts & Design
    "Creative Thinking", "Art History", "Painting", "Sculpture", "Photography",
    "Graphic Design", "Adobe Creative Suite", "Illustration", "Printmaking", "Visual Arts Theory",
    "Digital Art", "Typography", "Branding", "Animation", "Color Theory",
    "Storyboarding", "UI Design", "Motion Graphics", "Portfolio Development", "Design Thinking",

    # Software Testing & Automation
    "Selenium", "Test Automation Frameworks", "Java", "Jenkins", "TestNG",
    "Cucumber", "Python", "CI/CD", "Unit Testing", "Performance Testing",
    "Postman", "REST Assured", "API Testing", "Regression Testing", "Load Testing",
    "Bug Tracking", "Test Strategy", "Agile Testing", "Test Planning", "Test Reporting",

    # Blockchain & Web3
    "Blockchain Fundamentals", "Ethereum", "Solidity", "Smart Contracts", "Cryptography",
    "Decentralized Applications (dApps)", "Web3.js", "IPFS", "Ethereum Virtual Machine", "Blockchain Security",
    "Tokenomics", "Truffle Suite", "Hardhat", "Consensus Algorithms", "Gas Optimization",
    "DeFi Protocols", "Blockchain Governance", "NFT Standards", "Chainlink", "Layer 2 Solutions",

    # Business Analysis & Project Management
    "Requirement Gathering", "SQL", "Business Process Modeling", "Data Analysis", "Agile Methodology",
    "Stakeholder Management", "Project Management", "Excel", "Risk Management", "Documentation",
    "Business Intelligence", "Use Case Development", "KPI Tracking", "Process Reengineering", "Gap Analysis",
    "User Stories", "Scrum", "JIRA", "Feasibility Analysis", "SWOT Analysis",

    # Civil Engineering
    "Structural Analysis", "Construction Management", "AutoCAD", "Project Management", "Soil Mechanics",
    "Surveying", "Concrete Design", "Transportation Engineering", "Materials Science", "Environmental Engineering",
    "Geotechnical Engineering", "Urban Planning", "Hydraulics", "Building Codes", "Site Development",
    "BIM", "Construction Estimation", "Reinforced Concrete Design", "Steel Structures", "Construction Safety",

    # Data Science & Machine Learning
    "Python", "Machine Learning", "Data Visualization", "SQL", "Deep Learning",
    "Big Data", "Statistics", "Pandas", "Numpy", "Data Preprocessing",
    "Scikit-learn", "TensorFlow", "Keras", "Time Series Analysis", "Model Evaluation",
    "Feature Engineering", "Clustering", "Natural Language Processing", "Reinforcement Learning", "Computer Vision",

    # Database Engineering
    "SQL", "Database Design", "Data Modeling", "MySQL", "Oracle",
    "NoSQL", "Database Optimization", "ETL Processes", "Database Security", "Backup and Recovery",
    "PostgreSQL", "MongoDB", "Database Indexing", "Stored Procedures", "Replication",
    "Sharding", "Data Warehousing", "DBA Tasks", "Query Optimization", "OLAP",

    # DevOps & Cloud
    "CI/CD", "Docker", "Kubernetes", "AWS", "Azure",
    "Jenkins", "Ansible", "Terraform", "Git", "Linux/Unix",
    "Prometheus", "Grafana", "Helm", "CloudFormation", "Load Balancing",
    "Monitoring Tools", "Log Aggregation", "Containerization", "Cloud Security", "Serverless",

    # Microsoft Stack (.NET)
    "C#", "ASP.NET", "MVC", "SQL Server", "Web API",
    "Entity Framework", "LINQ", "JavaScript", "HTML/CSS", "Visual Studio",
    "Blazor", "XAML", "WPF", "WinForms", ".NET Core",
    "Azure DevOps", "NuGet", "Dependency Injection", "Middleware", "Unit Testing in .NET",

    # ETL & Data Warehousing
    "ETL Tools", "SQL", "Data Warehousing", "Data Integration", "Informatica",
    "Talend", "Data Transformation", "API Integration", "Scripting", "Big Data Technologies",
    "SSIS", "Airflow", "Snowflake", "Data Lake", "Data Pipeline Design",
    "Metadata Management", "Data Governance", "Star Schema", "Data Quality", "Batch Processing",

    # Electrical Engineering
    "Circuit Design", "MATLAB", "Power Systems", "Control Systems", "Signal Processing",
    "Power Electronics", "PCB Design", "Electrical Testing", "Project Management", "Energy Systems",
    "Microcontrollers", "Digital Electronics", "Embedded Systems", "Renewable Energy", "Switchgear",
    "High Voltage Engineering", "Simulation Tools", "SCADA", "Electrical CAD", "Transformer Design",

    # Human Resources (HR)
    "Recruitment", "Employee Relations", "HR Policies", "Payroll Management", "Employee Engagement",
    "Training and Development", "HR Software", "Compensation and Benefits", "Compliance", "Conflict Resolution",
    "Talent Management", "Succession Planning", "Workforce Planning", "Performance Appraisals", "HR Analytics",
    "Onboarding", "Labor Laws", "Benefits Administration", "Exit Interviews", "Employee Satisfaction",

    # Big Data Technologies
    "HDFS", "MapReduce", "Hive", "Pig", "Spark",
    "YARN", "HBase", "Big Data Technologies", "ETL", "Data Processing",
    "Apache Flink", "Apache Kafka", "Data Lakes", "Cluster Management", "Data Streaming",
    "Real-Time Processing", "Zookeeper", "Sqoop", "Presto", "Data Lineage",

    # Health & Fitness
    "Nutrition", "Exercise Science", "Personal Training", "Fitness Assessment", "Strength Training",
    "Cardiovascular Training", "Weight Loss Programs", "Health Coaching", "Anatomy", "Health and Wellness",
    "Sports Psychology", "Injury Prevention", "Rehabilitation", "Meal Planning", "Body Composition Analysis",
    "Wellness Programs", "Kinesiology", "Flexibility Training", "Group Fitness", "Mindfulness",

    # Java Development
    "Java", "Spring Framework", "Hibernate", "REST APIs", "SQL",
    "JUnit", "Maven/Gradle", "Multi-threading", "Object-Oriented Programming", "Agile Methodology",
    "JavaFX", "JSP/Servlets", "Spring Boot", "Dependency Injection", "Streams API",
    "Lambda Expressions", "Garbage Collection", "Collections Framework", "Thread Pooling", "JDBC",

    # Mechanical Engineering
    "CAD Software", "Thermodynamics", "Mechanical Design", "Material Science", "Finite Element Analysis",
    "Manufacturing Processes", "Mechanical Testing", "Project Management", "Quality Control", "Automated Systems",
    "Mechatronics", "HVAC", "Pneumatics", "Kinematics", "3D Printing",
    "CAM", "CNC Machines", "Vibration Analysis", "Stress Analysis", "Technical Drawing",

    # Cybersecurity & Networking
    "Network Protocols", "Firewalls", "VPN", "Security Audits", "Encryption",
    "IDS/IPS", "TCP/IP", "Risk Assessment", "Cybersecurity", "Penetration Testing",
    "Ethical Hacking", "Incident Response", "SOC Tools", "Vulnerability Assessment", "SIEM",
    "Malware Analysis", "Zero Trust Architecture", "Phishing Detection", "Network Hardening", "Access Control",

    # General Project Management
    "Project Management", "Supply Chain Management", "Budgeting", "Risk Management", "Team Leadership",
    "Process Optimization", "Quality Control", "Vendor Management", "Inventory Management", "Strategic Planning",
    "MS Project", "Agile Tools", "Waterfall Model", "Team Collaboration", "Time Management",
    "Project Documentation", "Milestone Tracking", "Project Lifecycle", "RACI Matrix", "Change Control",

    # Agile PM & Resource Mgmt
    "Project Management", "Project Scheduling", "Risk Management", "Agile Methodology", "Resource Allocation",
    "Stakeholder Management", "Project Planning", "Portfolio Management", "Cost Control", "Change Management",
    "Sprint Planning", "Kanban", "Scrum Master", "Agile Coaching", "Velocity Tracking",
    "Burndown Charts", "Daily Standups", "Agile Metrics", "Product Backlog", "Retrospectives",

    # Python Full Stack
    "Python", "Django", "Flask", "REST APIs", "SQL",
    "Unit Testing", "Git", "Pandas", "Numpy", "Data Structures",
    "FastAPI", "Jinja2", "ORM", "JWT Authentication", "API Rate Limiting",
    "CRUD Operations", "WebSockets", "Pagination", "Caching", "SQLite",

    # SAP Technologies
    "SAP ERP", "ABAP", "SAP Fiori", "SAP UI5", "SQL",
    "HANA", "NetWeaver", "SAP Cloud Platform", "SAP S/4HANA", "Business Process Integration",
    "BAPI", "IDoc", "SAP MM", "SAP SD", "SAP BW",
    "SAP CRM", "SAP Basis", "SAP Workflow", "FICO", "SAP Security",

    # Sales & Marketing
    "Sales Strategy", "CRM Software", "Negotiation", "Lead Generation", "Cold Calling",
    "Closing Deals", "Market Research", "Sales Presentations", "Customer Relationship Management", "Sales Forecasting",
    "Email Marketing", "B2B Sales", "Digital Marketing", "Sales Pipeline", "Account Management",
    "Inside Sales", "Customer Acquisition", "Upselling", "Sales Automation", "Competitive Analysis",

    # QA & Manual Testing
    "Manual Testing", "Automated Testing", "Selenium", "JUnit", "Bug Tracking Tools",
    "Test Plans", "Regression Testing", "Performance Testing", "API Testing", "Agile Methodologies",
    "Test Execution", "Exploratory Testing", "Smoke Testing", "Sanity Testing", "Defect Reporting",
    "Test Case Design", "Test Data Management", "Test Metrics", "Functional Testing", "Cross-Browser Testing",

    # Web Development & UI/UX
    "HTML/CSS", "JavaScript", "UX/UI Design", "Responsive Design", "Adobe Photoshop",
    "Wireframing", "SEO", "Bootstrap", "Web Accessibility", "Version Control",
    "React.js", "Vue.js", "Figma", "Adobe XD", "SASS/LESS",
    "User Flows", "Accessibility Testing", "Frontend Optimization", "Mobile-first Design", "Design Systems"
]

patterns = [nlp.make_doc(skill.lower()) for skill in skills_list]
matcher.add("SKILLS", None, *patterns)

def extract_custom_skills_using_phrasematcher(resume_text):
    # Process the resume text
    doc = nlp(resume_text.lower())
    
    # Apply the PhraseMatcher to the processed document
    matches = matcher(doc)
    
    # Extract the matched skills
    extracted_skills = set([doc[start:end].text for match_id, start, end in matches])
    
    # Return skills as a comma-separated list
    return ", ".join(extracted_skills)