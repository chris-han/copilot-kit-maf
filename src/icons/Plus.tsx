import { Plus as LucidePlus, LucideProps } from 'lucide-react';

const Plus = ({ className, ...props }: LucideProps) => {
  return <LucidePlus className={className} {...props} />;
};

export default Plus;