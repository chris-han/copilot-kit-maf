import { FileText as LucideFileText, LucideProps } from 'lucide-react';

const Docs = ({ className, ...props }: LucideProps) => {
  return <LucideFileText className={className} {...props} />;
};

export default Docs;